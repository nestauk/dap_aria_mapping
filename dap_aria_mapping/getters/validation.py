from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME
from typing import Dict

def get_tree_depths(tax_name: str) -> Dict[str, int]:
    """gets a dictionary with metrics about the tree depth of the taxonomies

    Args:
        tax_name (str): name of taxonomy to load results. Options are: cooccur, centroids, imbalanced 

    Returns:
        dict: size and depth metrics of trees in taxonomy
    """
    return download_obj(
        BUCKET_NAME,
        'outputs/validation_metrics/taxonomy_depth/{}.json'.format(tax_name),
        download_as = "dict")