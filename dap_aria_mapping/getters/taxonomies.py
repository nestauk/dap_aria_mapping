from dap_aria_mapping import BUCKET_NAME, PROJECT_DIR
from typing import Dict
from nesta_ds_utils.loading_saving.S3 import download_obj
import boto3, yaml, pickle
import pandas as pd
from toolz import pipe


def get_entity_embeddings(discarded: bool = False) -> pd.DataFrame:
    """Downloads entity embeddings from S3 and returns them as a pandas dataframe

    Args:
        discarded (bool, optional): Whether to include discarded entities. Defaults to False.

    Returns:
        pd.DataFrame: Embeddings dataframe
    """
    if not discarded:
        obj_pickle = "embeddings"
    else:
        obj_pickle = "embeddings_discarded"

    s3 = boto3.client("s3")
    embeddings_object = s3.get_object(
        Bucket=BUCKET_NAME, Key=f"outputs/embeddings/{obj_pickle}.pkl"
    )
    embeddings = pickle.loads(embeddings_object["Body"].read())
    embeddings = pd.DataFrame.from_dict(embeddings).T
    return embeddings


def get_cooccurrences(
    cluster_object: str = "meta_cluster", bucket_name: str = BUCKET_NAME
) -> pd.DataFrame:
    """Downloads cooccurrences from S3 and returns them as a pandas dataframe.
    The possible objects are:
        - meta_cluster
        - imbalanced
        - centroids

    Returns:
        pd.DataFrame: Cooccurrences dataframe
    """
    s3 = boto3.client("s3")
    meta_cluster_s3 = s3.get_object(
        Bucket=bucket_name,
        Key=f"outputs/semantic_taxonomy/cooccurrences/{cluster_object}.parquet",
    )

    return pipe(meta_cluster_s3["Body"], lambda x: pd.read_parquet(x))


def get_taxonomy_config() -> dict:
    """gets config with parameters used to generate taxonomy

    Returns:
        dict: config for taxonomy development
    """
    with open(f"{PROJECT_DIR}/dap_aria_mapping/config/taxonomy.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return config


def get_cooccurrence_taxonomy(sample: int = None) -> pd.DataFrame:
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
    return download_obj(
        BUCKET_NAME,
        "outputs/community_detection_taxonomy/tax.parquet",
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
        "outputs/community_detection_taxonomy/test.parquet",
        download_as="dataframe",
    )


def get_semantic_taxonomy(
    cluster_object: str = "centroids", sample: int = None
) -> pd.DataFrame:
    """Downloads taxonomy from S3 and returns them as a pandas dataframe.

    Args:
        cluster_object (str, optional): The type of semantic cluster object to download.
            Defaults to "centroids".
        sample (int, optional): The size of the sample to download - must be one of the option
            that were run. (i.e. 10, 25, 50, 75, 90). If None, downloads full taxonomy. Defaults to None.

    Returns:
        pd.DataFrame: A pandas dataframe containing the semantic taxonomy.
    """
    if sample:
        return download_obj(
            BUCKET_NAME,
            f"outputs/simulations/sample/formatted_outputs/semantic_{cluster_object}_clusters_sample_{str(sample)}.parquet",
            download_as="dataframe",
        )

    else:
        return download_obj(
            BUCKET_NAME,
            f"outputs/semantic_taxonomy/assignments/semantic_{cluster_object}_clusters.parquet",
            download_as="dataframe",
        )


def get_topic_names(
    taxonomy_class: str,
    name_type: str,
    level: int,
    long: bool = False,
    n_top: int = None,
) -> Dict[str, str]:
    """Downloads topic names from S3 and returns them as a dictionary.

    Args:
        taxonomy_class (str): The type of taxonomy to download.
        name_type (str): The type of name to download.
        level (int): The level of the taxonomy to download.
        n_top (int, optional): The number of top entities used to label. Defaults to None (10).
            chatgpt name_type uses 35 entities per topic to hit the API endpoint.
        long (bool, optional): Whether to download long names. Defaults to False.

    Returns:
        pd.DataFrame: A dictionary containing the topic names.
    """
    if long:
        if n_top is not None:
            return download_obj(
                BUCKET_NAME,
                f"outputs/topic_names/long_names/class_{taxonomy_class}_nametype_{name_type}_top_{str(n_top)}_level_{str(level)}.json",
                download_as="dict",
            )
        else:
            return download_obj(
                BUCKET_NAME,
                f"outputs/topic_names/long_names/class_{taxonomy_class}_nametype_{name_type}_level_{str(level)}.json",
                download_as="dict",
            )
    if n_top is not None:
        return download_obj(
            BUCKET_NAME,
            f"outputs/topic_names/class_{taxonomy_class}_nametype_{name_type}_top_{str(n_top)}_level_{str(level)}.json",
            download_as="dict",
        )
    else:
        return download_obj(
            BUCKET_NAME,
            f"outputs/topic_names/class_{taxonomy_class}_nametype_{name_type}_level_{str(level)}.json",
            download_as="dict",
        )