import os
import pandas as pd
import numpy as np
from scipy import stats
import json
import math

TRAITS = ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']
TRAIT_CODES = {
    "Agreeableness": ["agr", "ant"],
    "Conscientiousness": ["con", "unc"],
    "Extraversion": ["ext", "int"],
    "Neuroticism": ["neu", "emo"],
    "Openness": ["ope", "clo"]
}

"""
Helper Fns
"""

# All score reader
def read_gender_runs_scores():
    INPUT = os.path.join(os.getcwd(), 'scores', "all_scores.csv")
    with open(INPUT, 'r') as f:
        dataframe = pd.read_csv(INPUT)
    return dataframe

def read_liwc_scores():
    INPUT = os.path.join(os.getcwd(), 'scores', 'liwc_gender.csv')
    with open(INPUT, 'r') as f:
        dataframe = pd.read_csv(INPUT)
    dataframe['personality'] = ['_'.join(i.split('_')[1:])[:-9] for i in list(dataframe['Filename'])]
    dataframe['gender'] = [i.split('_')[0] for i in list(dataframe['Filename'])]
    return dataframe

# Helpers for statistical analysis
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
Stats Analysis
"""
def personality_t_test():
    df = read_gender_runs_scores()
    for trait_name in TRAITS:
        print("Trait name: ", trait_name)
        list1, list2 = get_all_participant(df, trait_name)
        score_1 = list(df[(df['personality_type'].isin(list1)) & (df["trait"] == trait_name)]["sum_score"])
        score_2 = list(df[(df['personality_type'].isin(list2)) & (df["trait"] == trait_name)]["sum_score"])
        print(np.mean(score_1), np.std(score_1), len(score_1))
        print(np.mean(score_2), np.std(score_2), len(score_2))

        N1 = single_variable_normality_test(score_1)
        N2 = single_variable_normality_test(score_2)
        if N1 & N2:
            one_way_anova(score_1, score_2)
        else:
            mannwhiteu_test(score_1, score_2)
        cohend = cohens_d(score_1, score_2)
        print("Cohen's d ", cohend)
        print("\n")

def liwc_data_proc():
    liwc_file = read_liwc_scores()
    personality_scores = read_gender_runs_scores()
    # Get the big five scores to matching
    big_five_data = {
        "Agreeableness": [],
        "Extraversion": [],
        "Conscientiousness": [],
        "Neuroticism": [],
        "Openness": []
    }

    for index, row in liwc_file.iterrows():
        file_string_ls = row['Filename'][:-4].split('_')
        gender = file_string_ls[0]
        personality = '_'.join(file_string_ls[1:-1])
        run = file_string_ls[-1]
        
        person_score = personality_scores.loc[(personality_scores['personality_type'] == personality) & (personality_scores['gender'] == gender) & (personality_scores['run'] == run)]
        for per in big_five_data.keys():
            big_five_data[per].append(person_score[person_score['trait'] == per]['sum_score'].values[0])

    for per in big_five_data.keys():
        liwc_file[per] = big_five_data[per]

    # Get which liwc variables we'd like to do stats analysis on 
    colums = liwc_file.columns
    liwc_var = list(colums)
    exclu_var = ['Segment', 'Filename', 'personality', 'gender'] + list(big_five_data.keys())
    liwc_var = list(set(liwc_var) - set(exclu_var))
    return liwc_file, liwc_var

def gender_distribution_diff(liwc_file, liwc_var):
    print("Gender Distri ")
    gender_data = []
    for var in liwc_var:
        score_1 = list(liwc_file[liwc_file['gender'] == "male"][var])
        score_2 = list(liwc_file[liwc_file['gender'] == "female"][var])

        N1 = single_variable_normality_test(score_1)
        N2 = single_variable_normality_test(score_2)
        test = ""
        if N1 & N2:
            test = "anova"
            val, p = one_way_anova(score_1, score_2)
        else:
            test = "m-w-test"
            val, p = mannwhiteu_test(score_1, score_2)

        if p <= 0.05:
            cohend = cohens_d(score_1, score_2)
        else:
            cohend = 0
        gender_data.append((var, p, cohend, np.mean(score_1), np.std(score_1), len(score_1), np.mean(score_2), np.std(score_2), len(score_2), val, test))
    cols = ["variable", "p-val", "cohensd", "male_mean", "male_std", "male_len",
            "female_mean", "female_std", "female_len",
            "effect", "test"]
    gender_df = pd.DataFrame(data=gender_data, columns=cols)
    gender_df.to_csv(os.path.join(os.getcwd(), "outputs", "stats", "gender_distri_cohend" + '.csv'))

def personality_liwc_stats(liwc_file, liwc_var):
    for trait_name in TRAITS:
        print("\nTrait name: ", trait_name)
        trait_data = liwc_file[trait_name].values
        trait_result = []
        for var in liwc_var:
            var_data = liwc_file[var].values
            stat, p = spearmanr(trait_data, var_data)
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
        trait_df.to_csv(os.path.join(os.getcwd(), "outputs", "stats", trait_name + '_corr.csv'))

def main():
    # On five dimensions, do t test for the scores with persona agents
    personality_t_test()

    if not os.path.isdir(os.path.join(os.getcwd(), "outputs", "stats")):
        os.mkdir(os.path.join(os.getcwd(), "outputs", "stats"))
    liwc_file, liwc_var = liwc_data_proc()
    # LIWC correlation with personality scores
    personality_liwc_stats(liwc_file, liwc_var)
    # LIWC and gender
    gender_distribution_diff(liwc_file, liwc_var)

if __name__ == "__main__":
    main()