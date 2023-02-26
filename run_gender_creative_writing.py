import argparse
from nltk.corpus import wordnet as wn
import nltk
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


def fill_the_prompt(bfi_question: str, gpt_answer: str, prompt_file=None):
    """construct the prompt from the prompt template"""
    prompt_template = open(prompt_file).read()
    prompt = prompt_template.replace("%BIGFIVE%", bfi_question)
    prompt = prompt.replace("%ANSWER%", gpt_answer)
    prompt = prompt.strip("\n").strip()
    prompt = prompt + "\n\n"
    return prompt

def write_bfi_story(json_filepath=None, prompt_file=None, output_folder=None):
    """"write one story given its bfi persona"""
    with open(json_filepath) as f:
        json_obj = json.load(f)

    persona_type = json_obj['persona_type']
    filled_prompt = fill_the_prompt(json_obj['prompt'], json_obj['text'], prompt_file)

    json_output = {"persona_type": persona_type, "text": [], "prompt": filled_prompt}

    answer = run_completion_query(filled_prompt)
    answer_text = answer['choices'][0]['text']
    json_output['text'] = answer_text
    return json_output


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, default="./prompts/bfi_writing_prompt.txt")
    parser.add_argument("--input_folder", default="./outputs/gender_bfi/temp0.7", type=str)
    parser.add_argument("--output_folder", default="./outputs/gender_bfi/temp0.7/writings", type=str)
    args = parser.parse_args()

    # create if not exits
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_folder, "male")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_folder, "female")).mkdir(parents=True, exist_ok=True)

    for run_id in range(1, 6):
        run_folder = "run" + str(run_id)
        run_folder_path = os.path.join(args.input_folder, run_folder)

        print(run_folder_path)
        # query for each persona type
        personality_types = [["extroverted", "introverted"],
                        ["agreeable", "antagonistic"],
                        ["conscientious", "unconscientious"],
                        ["neurotic", "emotionally stable"],
                        ["open to experience", "closed to experience"]]
    
        for persona_type in tqdm(list(itertools.product(*personality_types))):
            print("Processing persona --> {}".format(" + ".join(persona_type)))
            for gender in ['male', 'female']:
                persona_encoding = "_".join([trait[:3] for trait in persona_type])
                json_filepath = os.path.join(run_folder_path, gender, "{}.json".format(persona_encoding))
                json_output = write_bfi_story(json_filepath, prompt_file=args.prompt_file)
                # print(json_output)
                # save results in json
                json_output = json.dumps(json_output, indent=4)
                with open(os.path.join(args.output_folder, gender, "{}_{}.json".format(persona_encoding, run_folder)), "w") as out:
                    out.write(json_output)

if __name__ == "__main__":
    main()