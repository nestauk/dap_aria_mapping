from dap_aria_mapping.getters.validation import get_entity_groups
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
    entity_groupings: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, Dict[str, int]]:
    """calculate the average pairwise depth of a given taxonomy for all disciplines 

    Args:
        tax (pd.DataFrame): taxonomy
        entity_groupings (Dict[str, Dict[str, List[str]]]): config file with known entity groupings

    Returns:
        Dict[str, Dict[str, int]]: key: topic, value: key: discipline, value: average pairwise depth in taxonomy
    """

    output = defaultdict(lambda:defaultdict(int))
    for topic, discipline_data in entity_groupings.items():
            for discipline, entities in discipline_data.items():
                pairs = list(combinations(entities, 2))
                average_depth = average_pairwise_depth(tax, pairs)
                output[topic][discipline] = average_depth

    return output
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Taxonomy", formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--taxonomy', required = True, nargs="+",
                    help='name of the taxonomy/ies to analyse. Available optons are: cooccur, centroids, imbalanced, a list of multiple, or all')
    
    args = parser.parse_args()

    if args.taxonomy == ['all']:
        taxonomies = ['cooccur', 'centroids', 'imbalanced']
    else:
        taxonomies = args.taxonomy
    
    #load known groupings of entities to evaluate
    entity_groupings = get_entity_groups()

    if 'cooccur' in taxonomies:
        cooccurence_taxonomy = get_cooccurrence_taxonomy()
        output = generate_pairwise_output(cooccurence_taxonomy, entity_groupings)
        upload_obj(
            output,
            BUCKET_NAME,
            'outputs/validation_metrics/pairwise_depths/cooccur.json'
        )
    
    if 'centroids' in taxonomies:
        centroids_taxonomy = get_semantic_taxonomy(cluster_object='centroids')
        output = generate_pairwise_output(centroids_taxonomy, entity_groupings)
        upload_obj(
            output,
            BUCKET_NAME,
            'outputs/validation_metrics/pairwise_depths/centroids.json'
        )
    
    if 'imbalanced' in taxonomies:
        imbalanced_taxonomy = get_semantic_taxonomy(cluster_object='imbalanced')
        output = generate_pairwise_output(imbalanced_taxonomy, entity_groupings)
        upload_obj(
            output,
            BUCKET_NAME,
            'outputs/validation_metrics/pairwise_depths/imbalanced.json'
        )
