import argparse
import random
import json
import openai
import os
import pandas as pd
import sys
from tqdm import tqdm
import re
from pathlib import Path
import json
import numpy as np
import itertools
from gpt import run_completion_query
from tenacity import retry, stop_after_attempt, wait_random_exponential
import multiprocessing

openai.organization = ""
openai.api_key = ""

def construct_big_five_words(persona_type: list):
    """Construct the list of personality traits

    e.g., introverted + antagonistic + conscientious + emotionally stable + open to experience
    """
    options = list(persona_type)
    last_item = "and " + options[-1]
    options[-1] = last_item
    return ", ".join(options)

def run_gpt_query(model_name, temperature, system_prompt, prev_user_prompt, prev_assistant_prompt, user_prompt):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prev_user_prompt},
            {"role": "assistant", "content": prev_assistant_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
    )
    return response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))
def generate_bfi_story(model_name, temperature, persona_type, prompt_file, json_filepath):
    """Explains a given text for a specific audience.

    Args:
        text (str): The input text to be explained.
        prompt_file (str): The file path to the prompt file.

    Returns:
        str: The explanation of the input text.

    """
    # read previous json file
    with open(json_filepath) as f:
        json_obj = json.load(f)

    # Read prompt template
    system_prompt = "You are a character who is {}.".format(construct_big_five_words(persona_type))
    prev_user_prompt = open(prompt_file).read().strip("\n").strip() + "\n\n"
    prev_assistant_prompt = json_obj['annotation']

    user_prompt = "Please share a personal story in 800 words. Do not explicitly mention your personality traits in the story."
    user_prompt = user_prompt.strip("\n").strip()
    user_prompt = user_prompt + "\n\n"

    response = run_gpt_query(model_name, temperature, system_prompt, prev_user_prompt, prev_assistant_prompt, user_prompt)
    response = response["choices"][0]['message']['content'].strip("\n")
    return response, [system_prompt, prev_user_prompt, prev_assistant_prompt, user_prompt]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, default="./prompts/bfi_prompt.txt")
    parser.add_argument("--model", default="gpt-3.5-turbo-0613", type=str)
    parser.add_argument("--temperature", default=0.7, type=int)
    args = parser.parse_args()

    # create if not exits
    args.input_folder = os.path.join("./outputs/", args.model, "temp{}".format(args.temperature), "bfi")
    args.output_folder = os.path.join("./outputs/", args.model, "temp{}".format(args.temperature), "writing")
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    # query for each persona type
    personality_types = [["extroverted", "introverted"],
                        ["agreeable", "antagonistic"],
                        ["conscientious", "unconscientious"],
                        ["neurotic", "emotionally stable"],
                        ["open to experience", "closed to experience"]]
    
    pool = multiprocessing.Pool()
    responses = []

    for persona_type in tqdm(list(itertools.product(*personality_types))):
        for iteration in range(1, 11):
            persona_encoding = "_".join([trait[:4] for trait in persona_type])
            json_filename = os.path.join(args.input_folder, "{}_p{}.json".format(persona_encoding, iteration))
            
            # json_output = generate_bfi_story(args.model, args.temperature, persona_type, args.prompt_file, json_filename)
            # print(json_output)
            response = pool.apply_async(generate_bfi_story, args=(args.model, args.temperature, persona_type, args.prompt_file, json_filename))
            responses.append([persona_encoding, iteration, response])
            # print(response)
    
    for persona_encoding, iteration, response in tqdm(responses):
        json_obj = {"persona_encoding": persona_encoding, "iteration": iteration}
        json_obj["annotation"] = response.get()[0]
        json_obj["prev_prompts"] = response.get()[1]
        json_obj = json.dumps(json_obj, indent=4)
        with open(os.path.join(args.output_folder, "{}_p{}.json".format(persona_encoding, iteration)), "w", encoding='UTF-8') as out:
            out.write(json_obj)

if __name__ == "__main__":
    main()