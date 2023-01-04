from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME
import pandas as pd
import yaml


def get_taxonomy_config() -> dict:
    """gets config with parameters used to generate taxonomy

    Returns:
        dict: config for taxonomy development
    """
    with open("dap_aria_mapping/config/taxonomy.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return config


def get_cooccurrence_taxonomy() -> pd.DataFrame:
    """gets taxonomy developed using community detection on term cooccurrence network.
        Algorithm to generate taxonomy can be found in pipeline/taxonomy_development/community_detection.py.
        Parameters of taxonomy can be found in config/taxonomy.yaml.

    Returns:
        pd.DataFrame: table describing cluster assignments.
        Index: entity, Columns: levels of taxonomy, values are expressed as <INT>_<INT> where there is
        an integer to represent
    """
    return download_obj(
        BUCKET_NAME,
        "outputs/community_detection_clusters.parquet",
        download_as="dataframe",
    )


def get_test_cooccurrence_taxonomy() -> pd.DataFrame:
    """gets test taxonomy developed using community detection on term cooccurrence network.
        Algorithm to generate taxonomy can be found in pipeline/taxonomy_development/community_detection.py
        run in test_mode.
        Parameters of taxonomy can be found in config/taxonomy.yaml.

    Returns:
        pd.DataFrame: table describing cluster assignments.
        Index: entity, Columns: levels of taxonomy, values are expressed as <INT>_<INT> where there is
        an integer to represent
    """
    return download_obj(
        BUCKET_NAME,
        "outputs/test_community_detection_clusters.parquet",
        download_as="dataframe",
    )
