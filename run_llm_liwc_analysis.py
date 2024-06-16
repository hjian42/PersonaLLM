import os
import pandas as pd
import numpy as np
from scipy import stats
import json
import math

TRAITS = ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']
TRAIT_CODES = {
    "Agreeableness": ["agre", "anta"],
    "Conscientiousness": ["cons", "unco"],
    "Extraversion": ["extr", "intr"],
    "Neuroticism": ["neur", "emot"],
    "Openness": ["open", "clos"]
}

"""
Reader fns
"""
# All score reader
def read_personality_scores(writer):
    INPUT = os.path.join(os.getcwd(), 'scores', writer + "_all_scores.csv")
    with open(INPUT, 'r') as f:
        dataframe = pd.read_csv(INPUT)
    return dataframe

# liwc score reader: liwc table is generated from LIWC-22 software with the files in the folder "text"
def read_liwc_scores(writer):
    INPUT = os.path.join(os.getcwd(), 'liwc_rst', writer + '_liwc.csv')
    with open(INPUT, 'r') as f:
        dataframe = pd.read_csv(INPUT)
    dataframe['personality'] = ['_'.join(i.split('_')[1:])[:-9] for i in list(dataframe['Filename'])]
    return dataframe

"""
Process liwc file for stats analysis
label_type: # ["binary", "numeric"]
    - binary is converting the llm generated personality scores to binary labels, it is used for the comparative feature count with human stories in main paper
    - numeric is used for the liwc feature analysis in appendix
"""
def liwc_data_proc(writer, label_type="binary"):
    liwc_file = read_liwc_scores(writer)
    personality_scores = read_personality_scores(writer)
    # Get the big five scores to matching
    big_five_data = {
        "Agreeableness": [],
        "Extraversion": [],
        "Conscientiousness": [],
        "Neuroticism": [],
        "Openness": []
    }

    if label_type == "numeric":
        for index, row in liwc_file.iterrows():
            file_string_ls = row['Filename'][:-4].split('_')
            personality = '_'.join(file_string_ls[:-1])
            run = int(file_string_ls[-1].replace("p", ""))

            person_score = personality_scores.loc[(personality_scores['personality_type'] == personality) & (personality_scores['run'] == run)]
            for per in big_five_data.keys():
                big_five_data[per].append(person_score[person_score['trait'] == per]['sum_score'].values[0])

        for per in big_five_data.keys():
            liwc_file[per] = big_five_data[per]
    
    if label_type == "binary":
        personality_list = ['_'.join(item[:-4].split('_')[:-1]) for item in liwc_file['Filename']]
        for per in big_five_data.keys():
            big_five_data[per] = map_true_labels(personality_list, per)
            liwc_file[per] = big_five_data[per]

    # Get which liwc variables we'd like to do stats analysis on 
    liwc_var = list(liwc_file.columns)
    exclu_var = ['Segment', 'Filename', 'personality', 'gender'] + list(big_five_data.keys())
    liwc_var = list(set(liwc_var) - set(exclu_var))
    return liwc_file, liwc_var

def map_true_labels(persona_list, personality):
    personality_indices = {
        "Extraversion": {
            "index": 0,
            "label": {
                "intr": 0,
                "extr": 1
            }
        },
        "Agreeableness": {
            "index": 1,
            "label": {
                "anta": 0,
                "agre": 1
            }
        },
        "Conscientiousness": {
            "index": 2,
            "label": {
                "unco": 0,
                "cons": 1
            }
        },
        "Neuroticism": {
            "index": 3,
            "label": {
                "emot": 0,
                "neur": 1
            }
        },
        "Openness": {
            "index": 4,
            "label": {
                "clos": 0,
                "open": 1
            }
        }
    }
    idx = personality_indices[personality]['index']
    label_list = list(map(lambda x: personality_indices[personality]["label"][x.split('_')[idx]], persona_list))
    return label_list

def get_all_participant(df, persona_dim):
    
    """
    Pass in all score dataframe and get participants based on personality dimensions
    """
    big_five = dict()
    all_traits = list(set(df['personality_type']))
    print(all_traits)
    for trait in all_traits:
        five = trait.split("_")
        for i in five:
            if i not in big_five.keys():
                big_five[i] = []
            if trait not in big_five[i]:
                big_five[i].append(trait)
    return big_five[TRAIT_CODES[persona_dim][0]], big_five[TRAIT_CODES[persona_dim][1]]

"""
Helpers for statistical analysis
"""
def single_variable_normality_test(test_array):
    k, p = stats.normaltest(test_array)
    # threshold = 1e-2
    threshold = 0.05
    if p < threshold:
        return False
    else:
        return True

def mannwhiteu_test(data1, data2):
    """
    Mann White Test for samples from two distributions
    """
    threshold = 0.05
    _u1, _p = stats.mannwhitneyu(data1, data2, method="exact")
    if _p < threshold:
        print("MW: Reject Null Hypothesis In Favor of our hypothesis %f, %f"%(_u1, _p))
    return _u1, _p

def one_way_anova(data1, data2):
    """
    Mann White Test for samples from two distributions
    """
    threshold = 0.05
    _f, _p = stats.f_oneway(data1, data2)
    if _p < threshold:
        print("Anova: Reject Null Hypothesis In Favor of our hypothesis %f, %f"%(_f, _p))
    return _f, _p

def cohens_d(sample1, sample2):
    u1 = np.mean(sample1)
    u2 = np.mean(sample2)
    s1 = np.std(sample1)
    s2 = np.std(sample2)
    n1 = len(sample1)
    n2 = len(sample2)
    pooled = math.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    d = (u1 - u2)/pooled
    return round(d, 4)

def spearmanr(list1, list2):
    stat, p = stats.spearmanr(list1, list2)
    return stat, p

"""
Stats Analysis
"""
def personality_t_test(writer):
    print("Personality IBF T-Test for: ", writer)
    df = read_personality_scores(writer)

    for trait_name in TRAITS:
        print("Trait name: ", trait_name)
        list1, list2 = get_all_participant(df, trait_name)
        score_1 = list(df[(df['personality_type'].isin(list1)) & (df["trait"] == trait_name)]["mean_score"])
        score_2 = list(df[(df['personality_type'].isin(list2)) & (df["trait"] == trait_name)]["mean_score"])
        print(np.mean(score_1), np.std(score_1), len(score_1))
        print(np.mean(score_2), np.std(score_2), len(score_2))
        N1 = single_variable_normality_test(score_1)
        N2 = single_variable_normality_test(score_2)
        _v = None
        _p = None
        if N1 & N2:
            _v, _p = one_way_anova(score_1, score_2)
        else:
            _v, _p = mannwhiteu_test(score_1, score_2)
        cohend = cohens_d(score_1, score_2)
        print("Cohen's d ", cohend)
        print("\n")

def personality_liwc_stats(liwc_file, liwc_var, writer, label_type="binary"):
    print("Personality LIWC Test for: ", writer)
    for trait_name in TRAITS:
        print("\nTrait name: ", trait_name)
        trait_data = liwc_file[trait_name].values
        trait_result = []
        for var in liwc_var:
            var_data = liwc_file[var].values
            if label_type == "numeric":
                stat, p = spearmanr(trait_data, var_data)
            elif label_type == "binary":
                stat, p = stats.pointbiserialr(trait_data, var_data)
            else:
                raise Exception("WRONG Stats Type!!! ", label_type)
            pos = ""
            if p <= 0.05:
                if stat > 0:
                    pos = "Positive"
                elif stat == 0:
                    pos = "None"
                else:
                    pos = "Negative"
            trait_result.append((var, stat, p, pos, np.mean(var_data), np.std(var_data), len(var_data), np.mean(trait_data), np.std(trait_data), len(trait_data)))
            
        trait_col = ["Liwc", "statistics", "p-val", "rel", "liwc_mean", "liwc_std", "liwc_len", 
                    "trait_mean", "trait_std", "trait_len"] 

        trait_df = pd.DataFrame(data=trait_result, columns=trait_col)
        trait_df.to_csv(os.path.join(os.getcwd(), "stats_rst", label_type + "_" + writer + "_" + trait_name + '_corr.csv'))

def main():
    if not os.path.isdir(os.path.join(os.getcwd(), "stats_rst")):
        os.mkdir(os.path.join(os.getcwd(), "stats_rst"))

    writers = ["gpt3.5_turbo_0613", "gpt4_0613", "llama-2"]
    label_type = "binary" # numeric
    for writer in writers:
        personality_t_test(writer)

        liwc_file, liwc_var = liwc_data_proc(writer, label_type)
        # LIWC correlation with personality scores
        personality_liwc_stats(liwc_file, liwc_var, writer, label_type)

if __name__ == "__main__":
    main()