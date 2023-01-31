import argparse
from collections import defaultdict
from dap_aria_mapping.getters.taxonomies import get_cooccurrence_taxonomy, get_semantic_taxonomy
import pandas as pd
from scipy.stats import entropy
from nesta_ds_utils.loading_saving.S3 import upload_obj
from dap_aria_mapping import BUCKET_NAME
from itertools import repeat
from typing import Dict

def tax_entropy(tax: pd.DataFrame) -> Dict[str, int]:
    """calculates the chi square test statistic of the null hypothesis that the number of entities
    per taxonomy category follows a uniform distribution 

    Args:
        tax (pd.DataFrame): taxonomy
        tax_name (str): name of taxonomy
        metrics (dict): dictionary to store results

    Returns:
        dict: dictionary with results
    """

    metrics = defaultdict(int)

    for i in range(0, len(tax.columns)):
        dist = tax.iloc[:, i].value_counts()
        total_categories = len(dist)
        total_entities = sum(dist.values)
        pk = list(repeat((1/total_categories), times = total_categories))
        qk = [i/total_entities for i in dist.values]

        metrics["Level_{}".format(i+1)] = entropy(pk,qk)

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
    
    metrics = {tax: defaultdict(float) for tax in taxonomies}

    if 'cooccur' in taxonomies:
        cooccurence_taxonomy = get_cooccurrence_taxonomy()
        metrics = tax_entropy(cooccurence_taxonomy)
        upload_obj(
        metrics,
        BUCKET_NAME,
        'outputs/validation_metrics/entropy/cooccur.json'
    )
    
    if 'centroids' in taxonomies:
        centroids_taxonomy = get_semantic_taxonomy(cluster_object='centroids').set_index('tag')
        metrics = tax_entropy(centroids_taxonomy)
        upload_obj(
        metrics,
        BUCKET_NAME,
        'outputs/validation_metrics/entropy/centroids.json'
    )
    
    if 'imbalanced' in taxonomies:
        imbalanced_taxonomy = get_semantic_taxonomy(cluster_object='kmeans_strict_imb').set_index('tag')
        metrics = tax_entropy(imbalanced_taxonomy)
        upload_obj(
        metrics,
        BUCKET_NAME,
        'outputs/validation_metrics/entropy/imbalanced.json'
    )
    

    
    

    
