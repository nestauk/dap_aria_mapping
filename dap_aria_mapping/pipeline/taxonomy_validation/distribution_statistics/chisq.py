import argparse
from collections import defaultdict
from dap_aria_mapping.getters.taxonomies import get_cooccurrence_taxonomy, get_semantic_taxonomy
import pandas as pd
from scipy.stats import chisquare
from nesta_ds_utils.loading_saving.S3 import upload_obj
from dap_aria_mapping import BUCKET_NAME
from itertools import repeat
from typing import Dict

def tax_chisq(tax: pd.DataFrame) -> Dict[str, int]:
    """calculates the chi square test statistic of the null hypothesis that the number of entities
    per taxonomy category follows a uniform distribution 

    Args:
        tax (pd.DataFrame): taxonomy

    Returns:
        dict: dictionary with results
    """
    metrics = defaultdict(int)
    for i in range(0, len(tax.columns)):
        dist = tax.iloc[:, i].value_counts()
        total_categories = len(dist)
        total_entities = sum(dist.values)
        f_exp = list(repeat((total_entities/total_categories), times = total_categories))
        f_obs = list(dist.values)

        metrics["Level_{}".format(i+1)] = chisquare(f_obs,f_exp)[0]

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
        metrics = tax_chisq(cooccurence_taxonomy)
    
    if 'centroids' in taxonomies:
        centroids_taxonomy = get_semantic_taxonomy(cluster_object='centroids').set_index('tag')
        metrics = tax_chisq(centroids_taxonomy)
    
    if 'imbalanced' in taxonomies:
        imbalanced_taxonomy = get_semantic_taxonomy(cluster_object='kmeans_strict_imb').set_index('tag')
        metrics = tax_chisq(imbalanced_taxonomy)
    
    upload_obj(
        metrics,
        BUCKET_NAME,
        'outputs/validation_metrics/chisq.json'
    )

    
    

    
