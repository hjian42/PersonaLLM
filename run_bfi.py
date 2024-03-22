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
from gpt import is_answer_in_valid_form
from tenacity import retry, stop_after_attempt, wait_random_exponential
import multiprocessing
import replicate

openai.organization = ""
openai.api_key = ""


def construct_big_five_words(persona_type):
    """Construct the list of personality traits

    e.g., introverted + antagonistic + conscientious + emotionally stable + open to experience
    """
    options = list(persona_type)
    last_item = "and " + options[-1]
    options[-1] = last_item
    return ", ".join(options)

def run_gpt_query(model_name, temperature, system_prompt, user_prompt):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response

def run_llama2_query(temperature, system_prompt, user_prompt):
    
    response = replicate.run(
    "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",
        input={"prompt": user_prompt,
               "system_prompt": system_prompt,
               "max_new_tokens": 500,
               "temperature": temperature,
               "top_p": 1.0,
               }
    )
    return response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))
def generate_bfi_response(model_name, temperature, persona_type, prompt_file):
    """Generate a response using the specified model, temperature, persona type, and prompt file.

    Args:
        model_name (str): The name of the model to use for generating the response.
        temperature (float): The temperature parameter for controlling the randomness of the generated response.
        persona_type (str): The type of persona to use for generating the response.
        prompt_file (str): The file containing the prompt to use for generating the response.

    Returns:
        str: bfi response
        str: user prompt

    """
    # Read prompt template
    big_five_trait_str = construct_big_five_words(persona_type)
    system_prompt = "You are a character who is {}.".format(big_five_trait_str)

    user_prompt = open(prompt_file).read()
    user_prompt = user_prompt.strip("\n").strip()
    user_prompt = user_prompt + "\n\n"

    while True:
        if model_name.lower().startswith("gpt"):
            response = run_gpt_query(model_name, temperature, system_prompt, user_prompt)
            response = response["choices"][0]['message']['content'].strip("\n")
        elif model_name.lower().startswith("llama"):
            # print("llama-2 generating")
            # print(system_prompt, user_prompt)
            response = run_llama2_query(temperature, system_prompt, user_prompt)
            response = "".join([each for each in response]).strip("\n").strip()
            # print("YOUR RESPONSE:")
            # print(response)

        if is_answer_in_valid_form(response, 44):
            return response, user_prompt
        else:
            print("====>>> Answer not right, re-submitting request...")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, default="./prompts/bfi_prompt.txt")
    parser.add_argument("--model", default="gpt-3.5-turbo-0613", type=str)
    parser.add_argument("--temperature", default=0.7, type=float)
    args = parser.parse_args()

    # create if not exits
    args.output_folder = os.path.join("./outputs/", args.model, "temp{}".format(args.temperature), "bfi")
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
        print("Processing persona --> {}".format(" + ".join(persona_type)))
        for iteration in range(1, 11):
            # json_output = generate_bfi_response(args.model, args.temperature, persona_type, prompt_file=args.prompt_file)
            response = pool.apply_async(generate_bfi_response, args=(args.model, args.temperature, persona_type, args.prompt_file))
            persona_encoding = "_".join([trait[:4] for trait in persona_type])
            responses.append([persona_encoding, iteration, response])
            # print(json_output)
    
    for persona_encoding, iteration, response in tqdm(responses):
        json_obj = {"persona_encoding": persona_encoding, "iteration": iteration}
        json_obj["annotation"] = response.get()[0]
        json_obj["user_prompt"] = response.get()[1]
        json_obj = json.dumps(json_obj, indent=4)
        with open(os.path.join(args.output_folder, "{}_p{}.json".format(persona_encoding, iteration)), "w", encoding='UTF-8') as out:
            out.write(json_obj)

if __name__ == "__main__":
    main()