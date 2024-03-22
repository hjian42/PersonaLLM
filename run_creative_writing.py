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
# from gpt import run_completion_query
from tenacity import retry, stop_after_attempt, wait_random_exponential
import multiprocessing
import replicate

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

def run_llama2_query(temperature, system_prompt, prev_user_prompt, prev_assistant_prompt, user_prompt):
    system_prompt = "\n".join([system_prompt, prev_user_prompt, prev_assistant_prompt])
    response = replicate.run(
    "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",
        input={"prompt": user_prompt,
               "system_prompt": system_prompt,
               "max_new_tokens": 2000,
               "temperature": temperature,
               "top_p": 1.0,
               }
    )
    return response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))
def generate_bfi_story(model_name, temperature, persona_type, prompt_file, json_filepath):
    """Generate a story using a pre-trained language model with the specified parameters.

    Args:
        model_name (str): Name of the language model to use.
        temperature (float): Controls the randomness of the output.
        persona_type (str): Type of persona to use for generating the story.
        prompt_file (str): File containing a prompt for the story.
        json_filepath (str): Path to the JSON file to save the generated story.

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

    if model_name.lower().startswith("gpt"):
        response = run_gpt_query(model_name, temperature, system_prompt, prev_user_prompt, prev_assistant_prompt, user_prompt)
        response = response["choices"][0]['message']['content'].strip("\n")
    elif model_name.lower().startswith("llama"):
        response = run_llama2_query(temperature, system_prompt, prev_user_prompt, prev_assistant_prompt, user_prompt)
        response = "".join([each for each in response]).strip("\n").strip()
    return response, [system_prompt, prev_user_prompt, prev_assistant_prompt, user_prompt]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, default="./prompts/bfi_prompt.txt")
    parser.add_argument("--model", default="gpt-3.5-turbo-0613", type=str)
    parser.add_argument("--temperature", default=0.7, type=float)
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

    for persona_type in tqdm(list(itertools.product(*personality_types))[14:]):
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