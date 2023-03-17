from dap_aria_mapping import BUCKET_NAME, PROJECT_DIR
from typing import Dict
from nesta_ds_utils.loading_saving.S3 import download_obj
import boto3
import pandas as pd


def get_simulation_outputs(simulation: str = "sample") -> Dict[str, pd.DataFrame]:
    """gets taxonomy developed using community detection on term cooccurrence network.
        Algorithm to generate taxonomy can be found in pipeline/taxonomy_development/community_detection.py.
        Parameters of taxonomy can be found in config/taxonomy.yaml.

    Args:
     sample (int, optional): The size of the sample to download - must be one of the option
            that were run. (i.e. 10, 25, 50, 75, 90). If None, downloads full taxonomy. Defaults to None.
    Returns:
        pd.DataFrame: table describing cluster assignments.
        Index: entity, Columns: levels of taxonomy, values are expressed as <INT>_<INT> where there is
        an integer to represent
    """
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("aria-mapping")
    files_to_get = [
        file.key
        for file in bucket.objects.filter(
            Prefix=f"outputs/simulations/{simulation}/formatted_outputs/"
        )
        if file.key.endswith(".parquet")
    ]

    taxonomy = {}
    for file in files_to_get:
        taxonomy[file.split("/")[-1].split(".")[0]] = download_obj(
            BUCKET_NAME, file, download_as="dataframe"
        )
    return taxonomy


def get_simulation_topic_names(
    taxonomy_class: str, ratio: str, simulation: str, level: int = 1
) -> Dict[int, Dict[str, str]]:
    """Downloads topic names from S3 for a given simulation and returns them as a dictionary.

    Args:
        taxonomy_class (str): The type of taxonomy to download.
        ratio (str): The ratio of the simulation to download.
        simulation (str): The simulation type to download.
        level (int, optional): The level of the taxonomy to download. Defaults to 1.

    Returns:
        pd.DataFrame: A dictionary of dictionaries containing the topic names,
            for different levels of the taxonomy.
    """
    ratio_str = "ratio" if simulation == "noise" else "sample"
    return download_obj(
        BUCKET_NAME,
        f"outputs/simulations/{simulation}/names/class_{taxonomy_class}_{ratio_str}_{ratio}_level_{level}.json",
        download_as="dict",
    )


def get_simulation_topic_sizes(
    taxonomy_class: str, ratio: str, simulation: str
) -> Dict[int, Dict[str, str]]:
    """Gets the sizes of the topics for a given simulation.

    Args:
        taxonomy_class (str): The type of taxonomy to download.
        ratio (str): The ratio of the simulation to download.
        simulation (str): The simulation type to download.

    Returns:
        pd.DataFrame: A dataframe containing the topic names,
            and the number of entities in each topic.
    """
    ratio_str = "ratio" if simulation == "noise" else "sample"
    return download_obj(
        BUCKET_NAME,
        f"outputs/simulations/{simulation}/metrics/sizes/class_{taxonomy_class}_{ratio_str}_{ratio}.parquet",
        download_as="dataframe",
    )


def get_simulation_topic_distances(
    taxonomy_class: str, ratio: str, simulation: str
) -> Dict[int, Dict[str, str]]:
    """Gets the distances between the topic entities and the centroids for a given simulation.

    Args:
        taxonomy_class (str): The type of taxonomy to download.
        ratio (str): The ratio of the simulation to download.
        simulation (str): The simulation type to download.

    Returns:
        pd.DataFrame: A dataframe containing the distances between the topic entities and the centroids.

    """
    ratio_str = "ratio" if simulation == "noise" else "sample"
    return download_obj(
        BUCKET_NAME,
        f"outputs/simulations/{simulation}/metrics/distances/class_{taxonomy_class}_{ratio_str}_{ratio}.parquet",
        download_as="dataframe",
    )


def get_simulation_pairwise_combinations(
    taxonomy_class: str, ratio: str, simulation: str, topic_class: str = "topic"
) -> Dict[int, Dict[str, str]]:
    """Gets the pairwise combinations of pre-specified entities for a given simulation.

    Args:
        taxonomy_class (str): The type of taxonomy to download.
        ratio (str): The ratio of the simulation to download.
        simulation (str): The simulation type to download.

    Returns:
        pd.DataFrame: A dictionary of dictionaries containing the topic names,
            and the frequency with which pairs survive at each level.
    """
    assert topic_class in ["topic", "subtopic"]
    ratio_str = "ratio" if simulation == "noise" else "sample"
    return download_obj(
        BUCKET_NAME,
        f"outputs/simulations/{simulation}/metrics/pairwise/{topic_class}_class_{taxonomy_class}_{ratio_str}_{ratio}.json",
        download_as="dict",
    )
