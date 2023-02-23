
import openai
import re

openai.organization = ""
openai.api_key = ""

def run_completion_query(prompt):
    """Run a single prompt to GPT-3"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        stop=None,
        n=1,
        frequency_penalty=0.0,
        presence_penalty=0,
    )
    return response

def is_answer_in_valid_form(answer, num_sents):
    """Check if the GPT-3's answer is in the expected format.
    Enforcing the format makes it easy to extract sense predictions.
    
    This is the format we want:
        (a) 1
        (b) 2
        (c) 3
        ......
    """
    if "(a)" in answer:
        idx = answer.index("(a)")
    else:
        return False
    
    proc_answer = answer[idx:].strip("\n")
    tuples = proc_answer.split("\n")
    if len(tuples) != num_sents:
        return False
    else:
        for tup in tuples:
            tup = tup.strip()
            if len(tup.split()) != 2:
                return False
            if not re.search("\([a-z]+\) \d", tup):
                return False
        return True