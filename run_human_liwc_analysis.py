# Libraries
import os
import pandas as pd
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter

TRAITS = ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']

def get_personality_tag(row_data, total_personality):
    #cAGR  cCON  cEXT  cOPN  cNEU
    parse_tag = {
        "cEXT": ["intr", "extr"],
        "cAGR": ["anta", "agre"],
        "cCON": ["unco", "cons"],
        "cNEU": ["emot", "neur"],
        "cOPN": ["clos", "open"]
    }
    personality_type = []
    for key in parse_tag.keys():
        personality_type.append(parse_tag[key][int(row_data[key])])
    pers_str = '_'.join(personality_type)
    id = total_personality.count(pers_str)
    return pers_str + "_p" + str(id + 1)


def process_human_story_corpus():
    HUMAN_STORIES = os.path.join(os.getcwd(), "essays2007", "essays2007.csv")
    OUTPUT_HUMAN_TEXT_PATH = os.path.join(os.getcwd(), "text", "human_stories")
    if not os.path.isdir(OUTPUT_HUMAN_TEXT_PATH):
        os.mkdir(OUTPUT_HUMAN_TEXT_PATH)

    human_story = pd.read_csv(HUMAN_STORIES)
    # Process human story into text
    personality_list = []
    stories = []
    story_wc = []
    for index, row in human_story.iterrows():
        story = row['text']
        wc = len(story.split(" "))
        if wc < 10:
            # remove stories that have low word counts
            continue
        personality_str = get_personality_tag(row, personality_list)
        personality_list.append('_'.join(personality_str.split("_")[:-1]))
        stories.append(story)
        story_wc.append(wc)
        with open(os.path.join(OUTPUT_HUMAN_TEXT_PATH, personality_str + ".txt"), 'w') as f:
            f.write(story)

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

def main():
    # Process from human story corpus to liwc processing format
    # process_human_story_corpus()

    if not os.path.isdir(os.path.join(os.getcwd(), "stats_rst")):
        os.mkdir(os.path.join(os.getcwd(), "stats_rst"))

    # Processing liwc file to stats
    LIWC_FILE = os.path.join(os.getcwd(), "liwc_rst", "human_liwc.csv")
    liwc_file = pd.read_csv(LIWC_FILE)
    liwc_file['personality'] = ['_'.join(i.split('_')[:-1]) for i in list(liwc_file['Filename'])]
    # Map big five traits to binary bales
    big_five_data = {
        "Agreeableness": [],
        "Extraversion": [],
        "Conscientiousness": [],
        "Neuroticism": [],
        "Openness": []
    }
    personality_list = ['_'.join(item[:-4].split('_')[:-1]) for item in liwc_file['Filename']]
    for per in big_five_data.keys():
        # big_five_data[per] = map_true_labels(personality_list, per)
        liwc_file[per] = map_true_labels(personality_list, per)
    
    # Get which liwc variables we'd like to do stats analysis on 
    liwc_var = list(liwc_file.columns)
    exclu_var = ['Segment', 'Filename', 'personality', 'gender'] + list(big_five_data.keys())
    liwc_var = list(set(liwc_var) - set(exclu_var))
    for trait_name in TRAITS:
        print("\nTrait name: ", trait_name)
        trait_data = liwc_file[trait_name].values
        trait_result = []
        for var in liwc_var:
            var_data = liwc_file[var].values
            stat, p = stats.pointbiserialr(trait_data, var_data)
            pos = ""
            if p <= 0.05:
                if stat > 0:
                    pos = "Positive"
                elif stat == 0:
                    pos = "None"
                else:
                    pos = "Negative"
            trait_result.append((var, stat, p, pos, np.mean(var_data), np.std(var_data), len(var_data)))
            
        trait_col = ["Liwc", "statistics", "p-val", "rel", "liwc_mean", "liwc_std", "liwc_len"] 

        trait_df = pd.DataFrame(data=trait_result, columns=trait_col)
        trait_df.to_csv(os.path.join(os.getcwd(), "stats_rst", "binary_human_" + trait_name + '_corr.csv'))


if __name__ == "__main__":
    main()