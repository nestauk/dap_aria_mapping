from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME

def get_tree_depths() -> dict:
    """gets a dictionary with metrics about the tree depth of the taxonomies

    Returns:
        dict: size and depth metrics of trees in taxonomy
    """
    return download_obj(
        BUCKET_NAME,
        'outputs/validation_metrics/taxonomy_depth.json',
        download_as = "dict")