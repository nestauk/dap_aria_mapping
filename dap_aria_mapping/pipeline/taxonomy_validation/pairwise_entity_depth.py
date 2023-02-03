from dap_aria_mapping.getters.validation import get_topic_groups, get_subtopic_groups
from dap_aria_mapping.getters.taxonomies import get_cooccurrence_taxonomy, get_semantic_taxonomy
import argparse
import pandas as pd
from typing import List, Tuple, Dict
from collections import defaultdict
from itertools import combinations
from nesta_ds_utils.loading_saving.S3 import upload_obj
from dap_aria_mapping import BUCKET_NAME

def average_pairwise_depth(tax: pd.DataFrame, pairs: List[Tuple[str, str]]) -> int:
    """calculates the average level of the taxonomy that pairs in a list 
       stay together in the same group

    Args:
        tax (pd.DataFrame): taxonomy
        pairs (List[Tuple[str, str]]): tuple of pairs of entities

    Returns:
        int: average pairwise depth of the taxonomy
    """
    total = 0
    for pair in pairs:
        tax1 = list(tax.loc[pair[0]])
        tax2 = list(tax.loc[pair[1]])
        #BY CREATING A SET, WE ARE NOT DOUBLE COUNTING WHERE A TREE GETS CUT OFF AT A HIGHER LEVEL 
        total += len(set(tax1).intersection(tax2))
    return total/len(pairs)

def generate_pairwise_output(
    tax: pd.DataFrame, 
    entity_groupings: Dict[str, Dict[str, List[str]]],
    level: str
    ) -> Dict[str, Dict[str, int]]:
    """calculate the average pairwise depth of a given taxonomy for all disciplines 

    Args:
        tax (pd.DataFrame): taxonomy
        entity_groupings (Dict[str, Dict[str, List[str]]]): config file with known entity groupings
        level (str): level or resolution to test - either topics or subtopics

    Returns:
        Dict[str, Dict[str, int]]: key: topic, value: key: discipline, value: average pairwise depth in taxonomy
    """
    if level == 'topics':
        output = defaultdict(lambda:defaultdict(int))
        for area, discipline_data in entity_groupings.items():            
            for discipline, topics in discipline_data.items():
                pairs = list(combinations(topics, 2))
                average_depth = average_pairwise_depth(tax, pairs)
                output[area][discipline] = average_depth
    
    elif level == 'subtopics':
        output = defaultdict(lambda:defaultdict(lambda: defaultdict(int)))
        for area, discipline_data in entity_groupings.items():            
            for discipline, topic_data in discipline_data.items():
                for topic, subtopics in topic_data.items():
                    pairs = list(combinations(subtopics, 2))
                    average_depth = average_pairwise_depth(tax, pairs)
                    output[area][discipline][topic] = average_depth
                    
                

    return output
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--taxonomy', required = True, nargs="+",
                    help= 'name of the taxonomy/ies to analyse. Available optons are: cooccur, centroids, imbalanced, a list of multiple, or all')
    
    parser.add_argument('--level', required=True, 
                    help = 'level to test, options are: topics, subtopics')

    args = parser.parse_args()

    if args.taxonomy == ['all']:
        taxonomies = ['cooccur', 'centroids', 'imbalanced']
    else:
        taxonomies = args.taxonomy
    
    #load known groupings of entities to evaluate
    if args.level == 'topics':
        entity_groupings = get_topic_groups()
    elif args.level == 'subtopics':
        entity_groupings = get_subtopic_groups()
    else:
        print("Invalid argument passed as level")

    if 'cooccur' in taxonomies:
        cooccurence_taxonomy = get_cooccurrence_taxonomy()
        output = generate_pairwise_output(cooccurence_taxonomy, entity_groupings, args.level)
        if args.level == 'topics':
            upload_obj(
                output,
                BUCKET_NAME,
                'outputs/validation_metrics/pairwise_depths/topic_level/cooccur.json'
            )
        else:
            upload_obj(
                output,
                BUCKET_NAME,
                'outputs/validation_metrics/pairwise_depths/subtopic_level/cooccur.json'
            )
    
    if 'centroids' in taxonomies:
        centroids_taxonomy = get_semantic_taxonomy(cluster_object='centroids')
        output = generate_pairwise_output(centroids_taxonomy, entity_groupings, args.level)
        if args.level == 'topics':
            upload_obj(
                output,
                BUCKET_NAME,
                'outputs/validation_metrics/pairwise_depths/topic_level/centroids.json'
            )
        else:
            upload_obj(
                output,
                BUCKET_NAME,
                'outputs/validation_metrics/pairwise_depths/subtopic_level/centroids.json'
            )
    if 'imbalanced' in taxonomies:
        imbalanced_taxonomy = get_semantic_taxonomy(cluster_object='imbalanced')
        output = generate_pairwise_output(imbalanced_taxonomy, entity_groupings, args.level)
        if args.level == 'topics':
            upload_obj(
                output,
                BUCKET_NAME,
                'outputs/validation_metrics/pairwise_depths/topic_level/imbalanced.json'
            )
        else:
            upload_obj(
                output,
                BUCKET_NAME,
                'outputs/validation_metrics/pairwise_depths/subtopic_level/imbalanced.json'
            )
