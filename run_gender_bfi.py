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
from gpt import run_completion_query, is_answer_in_valid_form

def construct_big_five_words(persona_type: list):
    """Construct the list of personality traits

    e.g., introverted + antagonistic + conscientious + emotionally stable + open to experience
    """
    options = list(persona_type)
    last_item = "and " + options[-1]
    options[-1] = last_item
    return ", ".join(options)

def fill_the_prompt(gender: str, persona_type: list, prompt_file=None):
    """construct the prompt from the prompt template"""
    prompt_template = open(prompt_file).read()
    big_five_trait_str = construct_big_five_words(persona_type)
    prompt = prompt_template.replace("%MF%", gender)
    prompt = prompt.replace("%PERSONA%", big_five_trait_str)
    prompt = prompt.strip("\n").strip()
    prompt = prompt + "\n\n"
    return prompt

def answer_bfi_questions(gender, persona_type, prompt_file=None, output_folder=None):
    """"answer bfi questions for one persona"""
    filled_prompt = fill_the_prompt(gender, persona_type, prompt_file)

    json_output = {"persona_type": persona_type, "text": [], "prompt": filled_prompt}

    while True:
        answer = run_completion_query(filled_prompt)
        answer_text = answer['choices'][0]['text']
        if is_answer_in_valid_form(answer_text, 44):
            json_output['text'] = answer_text
            break
        else:
            print("====>>> Answer not right, re-submitting request...")
    return json_output


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, default="./prompts/bfi_gender_prompt.txt")
    parser.add_argument("--output_folder", default="./outputs/gender_bfi/temp0.7/run1", type=str)
    args = parser.parse_args()

    # create if not exits
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_folder, "male")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_folder, "female")).mkdir(parents=True, exist_ok=True)

    # query for each persona type
    personality_types = [["extroverted", "introverted"],
                        ["agreeable", "antagonistic"],
                        ["conscientious", "unconscientious"],
                        ["neurotic", "emotionally stable"],
                        ["open to experience", "closed to experience"]]
    
    for persona_type in tqdm(list(itertools.product(*personality_types))):
        print("Processing persona --> {}".format(" + ".join(persona_type)))
        for gender in ["male", "female"]:
            json_output = answer_bfi_questions(gender, persona_type, prompt_file=args.prompt_file)
            
            # save results in json
            if json_output['text']:
                json_output = json.dumps(json_output, indent=4)
                persona_encoding = "_".join([trait[:3] for trait in persona_type])
                with open(os.path.join(args.output_folder, gender, "{}.json".format(persona_encoding)), "w") as out:
                    out.write(json_output)

if __name__ == "__main__":
    main()