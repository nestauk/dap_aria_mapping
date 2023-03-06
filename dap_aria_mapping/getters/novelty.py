"""
Getters for novelty analyses results

"""
from dap_aria_mapping import logging, PROJECT_DIR, BUCKET_NAME
from nesta_ds_utils.loading_saving.S3 import download_obj
import pandas as pd


def get_openalex_novelty_scores(
    level: int = 5, from_local: bool = False
) -> pd.DataFrame:
    """
    Downloads novelty scores for OpenAlex papers

    Args:
        level (int): Taxonomy level that was used for novelty calculation
        from_local (bool): Whether to load the data from local disk

    Returns:
        pd.DataFrame: Novelty scores for OpenAlex papers. Columns are:
            - work_id: OpenAlex work id
            - year: publication year
            - commonness: commonness score for the paper
            - novelty_score: novelty score for the paper
            - n_topics: number of topics in the paper
            - topics: list of topics in the paper

    """
    filepath = f"outputs/novelty/openalex_novelty_{level}.parquet"
    if from_local:
        return pd.read_parquet(PROJECT_DIR / filepath)
    else:
        return download_obj(
            BUCKET_NAME,
            filepath,
            download_as="dataframe",
        )


def get_openalex_topic_pair_commonness(
    level: int = 5, from_local: bool = False
) -> pd.DataFrame:
    """
    Downloads topic pair commonness scores for OpenAlex papers

    Args:
        level (int): Taxonomy level that was used for commonness calculations
        from_local (bool): Whether to load the data from local disk

    Returns:
        pd.DataFrame: Topic pair commonness scores for OpenAlex papers. Columns are:
            - topic_1: first topic in the pair
            - topic_2: second topic in the pair
            - year: publication year
            - N_ij_t: Number of co-occurrences of topic_1 and topic_2 in year t
            - commonness: commonness score for the topic pair in year t

    """
    filepath = f"outputs/novelty/openalex_topic_pair_commonness_{level}.parquet"
    if from_local:
        return pd.read_parquet(PROJECT_DIR / filepath)
    else:
        return download_obj(
            BUCKET_NAME,
            filepath,
            download_as="dataframe",
        )
