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


def get_semantic_taxonomy(cluster_object: str = "centroids") -> pd.DataFrame:
    """Downloads taxonomy from S3 and returns them as a pandas dataframe.

    Args:
        cluster_object (str, optional): The type of semantic cluster object to download.
            Defaults to "centroids".

    Returns:
        pd.DataFrame: A pandas dataframe containing the semantic taxonomy.
    """
    return download_obj(
        BUCKET_NAME,
        f"outputs/semantic_taxonomy/assignments/semantic_{cluster_object}_clusters.parquet",
        download_as="dataframe",
    )


def get_topic_names(
    taxonomy_class: str, name_type: str, level: int, large: bool = False
) -> dict:
    """Downloads topic names from S3 and returns them as a dictionary.

    Args:
        taxonomy_class (str): The type of taxonomy to download.
        name_type (str): The type of name to download.
        level (int): The level of the taxonomy to download.

    Returns:
        pd.DataFrame: A dictionary containing the topic names.
    """
    if large:
        return download_obj(
            BUCKET_NAME,
            f"outputs/topic_names/large_topic_names/class_{taxonomy_class}_nametype_{name_type}_level_{str(level)}.json",
            download_as="dict",
        )
    else:
        return download_obj(
            BUCKET_NAME,
            f"outputs/topic_names/class_{taxonomy_class}_nametype_{name_type}_level_{str(level)}.json",
            download_as="dict",
        )
