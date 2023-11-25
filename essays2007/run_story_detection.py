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
from tenacity import retry, stop_after_attempt, wait_random_exponential
import multiprocessing
from string import punctuation

openai.organization = ""
openai.api_key = ""

def is_answer_in_valid_form(answer):
    """
    Check if the GPT's answer is in the expected format.

    The expected format is one of the following:
    - "yes"
    - "no"

    Note: The function allows for case-insensitive matching and ignores leading/trailing whitespaces.

    Parameters:
    - answer (str): The GPT-generated answer to be checked for validity.

    Returns:
    - bool: True if the answer is in the expected format, False otherwise.

    Example:
    >>> is_answer_in_valid_form("Yes")   # Returns True
    >>> is_answer_in_valid_form("NO ")    # Returns True
    >>> is_answer_in_valid_form("Maybe")  # Returns False
    """
    answer = answer.strip("\n").strip().lower().strip(punctuation)
    if "yes" == answer:
        return True
    elif "no" == answer:
        return True
    return False


def run_gpt_query(model_name, temperature, system_prompt):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        temperature=temperature,
    )
    return response

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))
def generate_story_response(model_name, temperature, essay_text):
    """Predict if an essay contains a personal story
    """
    # Read prompt template
    system_prompt = """Does the following text contain a personal story?\nAnswer YES or NO. No explanation is needed.\nSTORY: {}\nAsnwer: """.format(essay_text)

    system_prompt = system_prompt.strip("\n").strip()

    while True:
        response = run_gpt_query(model_name, temperature, system_prompt)
        response = response["choices"][0]['message']['content'].strip("\n")

        if is_answer_in_valid_form(response):
            return response, system_prompt
        else:
            print("====>>> Answer not right, re-submitting request...")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-3.5-turbo-0613", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    args = parser.parse_args()

    # create if not exits
    args.output_folder = os.path.join("../outputs/", args.model, "temp{}".format(args.temperature), "essays2007")
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    pool = multiprocessing.Pool()
    responses = []

    df_essays = pd.read_csv("essays2007.csv").iloc[2201:]
    for row in df_essays.iterrows():
        row = row[1]
        # print(row.text)
        response = pool.apply_async(generate_story_response, args=(args.model, args.temperature, row.text))
        responses.append([row.Filename, response])
        # json_output = generate_story_response(args.model, args.temperature, row.text)
        # print(json_output)
        # break
    
    for file_name, response in tqdm(responses):
        json_obj = {"file_name": file_name}
        json_obj["annotation"] = response.get()[0]
        json_obj["user_prompt"] = response.get()[1]
        json_obj = json.dumps(json_obj, indent=4)
        with open(os.path.join(args.output_folder, "{}.json".format(file_name)), "w", encoding='UTF-8') as out:
            out.write(json_obj)

if __name__ == "__main__":
    main()