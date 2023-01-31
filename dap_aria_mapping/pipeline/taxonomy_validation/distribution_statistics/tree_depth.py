from dap_aria_mapping.getters.taxonomies import get_cooccurrence_taxonomy, get_semantic_taxonomy
import argparse
from sys import exit
from typing import Tuple
import pandas as pd
from nesta_ds_utils.loading_saving.S3 import upload_obj
from dap_aria_mapping import BUCKET_NAME
from collections import defaultdict

def _count_at_level(tax: pd.DataFrame(), level: int) -> Tuple[int]:
    """gets the counts of the unique topic groups at a given level of the taxonomy

    Args:
        tax (pd.DataFrame): taxonomy 
        level (int): level of the taxonomy, first level is 0

    Returns:
        Tuple[int]: count of unique values
    """
    return tax.iloc[:, level].nunique()

def _p_split(labels_at_level: set, labels_at_next_level: set) -> float:
    """gets the fraction of topics at a given level that split into subtopics at the next level

    Args:
        labels_at_level (set): unique topics at a given level
        labels_at_next_level (set): unique topics at the next level

    Returns:
        float: fraction of topics at a given level that split at the next level
    """
    return len([x for x in labels_at_level if x not in labels_at_next_level])/len(labels_at_level)

def add_tax_metrics(tax: pd.DataFrame) -> dict:
    """generates count of unique topics at each level, and percent of topics that split into subtopics at each level

    Args:
        tax (pd.DataFrame): taxonomy

    Returns:
        dict: dictionary with added metrics
    """
    metrics = defaultdict(int)
    for i, col in enumerate(tax.columns):
        #add the counts of unique values at each level of the taxonomy to the metrics
        #NOTE: given that the last level will carry down if a tree stops splitting, the bottom level contains some higer level groupings
        metrics["counts"]['{}'.format(i+1)] = _count_at_level(tax, i)
        #add the fraction of topics that split at a given level
        if i < len(tax.columns)-1:
            metrics["split_fractions"]['{}'.format(i+1)] = _p_split(set(tax.iloc[:, i]), set(tax.iloc[:, i+1]))

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Taxonomy", formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--taxonomy', required = True,
                    help='name of the taxonomy/ies to analyse. Available optons are: cooccur, centroids, imbalanced, a list of multiple, or all')
    
    args = parser.parse_args()

    if args.taxonomy == 'all':
        taxonomies = ['cooccur', 'centroids', 'imbalanced']
    else:
        taxonomies = list(args.taxonomy)
    

    if 'cooccur' in taxonomies:
        cooccurence_taxonomy = get_cooccurrence_taxonomy()
        metrics = add_tax_metrics(cooccurence_taxonomy)
        upload_obj(
            metrics,
            BUCKET_NAME,
            'outputs/validation_metrics/taxonomy_depth/cooccur.json'
        )

    
    if 'centroids' in taxonomies:
        centroids_taxonomy = get_semantic_taxonomy(cluster_object='centroids').set_index('tag')
        metrics = add_tax_metrics(centroids_taxonomy)
        upload_obj(
            metrics,
            BUCKET_NAME,
            'outputs/validation_metrics/taxonomy_depth/centroids.json'
        )
    
    if 'imbalanced' in taxonomies:
        imbalanced_taxonomy = get_semantic_taxonomy(cluster_object='kmeans_strict_imb').set_index('tag')
        metrics = add_tax_metrics(imbalanced_taxonomy)
        upload_obj(
            metrics,
            BUCKET_NAME,
            'outputs/validation_metrics/taxonomy_depth/imbalanced.json'
        )

    




    

