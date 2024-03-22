import itertools
from string import ascii_lowercase
import json
import pandas as pd

def iter_all_strings():
    """make a continuous alphabetic list python (from a-z then from aa, ab, ac etc)"""
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)


def main():
    # change the output folder
    model_name = "llama-2" # gpt-4-0613 or gpt-3.5-turbo-0613
    output_folder = "../outputs/{}/temp0.7/bfi".format(model_name)

    personality_types = [["extroverted", "introverted"],
                        ["agreeable", "antagonistic"],
                        ["conscientious", "unconscientious"],
                        ["neurotic", "emotionally stable"],
                        ["open to experience", "closed to experience"]]

    df_aggregated_scores = []
    for which_run in range(1, 11):
        run_name = "p{}".format(which_run)
        for persona_type in list(itertools.product(*personality_types)):

            persona_encoding = "_".join([trait[:4] for trait in persona_type])
            df_preds = pd.read_csv("../prompts/bfi_scores.txt", delimiter="\t")

            # read LLM response and compute the bfi scores
            with open("{}/{}_{}.json".format(output_folder, persona_encoding, run_name)) as f:
                json_obj = json.load(f)
                annotation = json_obj['annotation'][json_obj['annotation'].index("(a)"):].strip()
                predictions = [int(tup.split()[1]) for tup in annotation.split("\n")[:44]] # up to 44 BFI items
                df_preds['scores'] = predictions
                df_preds["scores"][df_preds.reverse == "R"] = df_preds["scores"][df_preds.reverse == "R"].map({1: 5, 2: 4, 3: 3, 4: 2, 5: 1})
                df_sum = df_preds[['trait', 'scores']].groupby(by="trait").sum().reset_index().sort_values(by=["trait"])
                df_mean = df_preds[['trait', 'scores']].groupby(by="trait").mean().reset_index().sort_values(by=["trait"])
                df_sum['run'] = which_run
                df_sum['personality_type'] = persona_encoding
                df_sum['sum_score'] = df_sum['scores']
                df_sum['mean_score'] = df_mean['scores']
                del df_sum['scores']
                df_aggregated_scores.append(df_sum)

    df_aggregated_scores = pd.concat(df_aggregated_scores)
    df_aggregated_scores.to_csv("./{}_all_scores.csv".format(model_name))
    print(df_aggregated_scores.head(5))


if __name__ == "__main__":
    main()