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
        level (int, optional): Taxonomy level that was used for novelty calculation. Defaults to 5.
        from_local (bool, optional): Whether to load the data from local disk. Defaults to False.

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
        level (int, optional): Taxonomy level that was used for commonness calculations. Defaults to 5.
        from_local (bool, optional): Whether to load the data from local disk. Defaults to False.

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


def get_topic_novelty_openalex(
    level: int = 5, from_local: bool = False
) -> pd.DataFrame:
    """
    Downloads topic-level novelty score (based on OpenAlex papers).
    There are two novelty scores: one based on aggregating document novelty scores,
    and one based on aggregating topic pair novelty scores.

    Note that for more granular levels, many of the topic pairs novelty scores
    might be missing if they didn't meet the minimum number of pair count threshold.
    To overcome this, one could use different pair count thresholds for different
    levels of the taxonomy.

    Args:
        level (int, optional): Taxonomy level that was used for commonness calculations. Defaults to 5.
        from_local (bool, optional): Whether to load the data from local disk. Defaults to False.

    Returns:
        pd.DataFrame: Table with topic-level novelty scores. Columns are:
            - topic: Topic identifier
            - topic_name: Human-readable name of the topic
            - year: Year to which the novelty score corresponds to
            - topic_doc_novelty: Novelty measure based on aggregating document novelty scores
            - doc_counts: Number of documents with the topic in the given year
            - topic_pair_novelty: Alternative novelty measure based on aggregating topic pair novelty scores
            - topic_pair_commonness: Aggregated topic commonness measure (ie, the inverse of topic pair novelty)
            - pair_counts: Number of topic pairs with the topic in the given year
    """
    filepath = f"outputs/novelty/topic_novelty_openalex_{level}.parquet"
    if from_local:
        return pd.read_parquet(PROJECT_DIR / filepath)
    else:
        return download_obj(
            BUCKET_NAME,
            filepath,
            download_as="dataframe",
        )


def get_patent_novelty_scores(level: int = 5, from_local: bool = False) -> pd.DataFrame:
    """
    Downloads novelty scores for patents

    Args:
        level (int): Taxonomy level that was used for novelty calculation
        from_local (bool): Whether to load the data from local disk

    Returns:
        pd.DataFrame: Novelty scores for patents. Columns are:
            - work_id: OpenAlex work id
            - year: priority year
            - commonness: commonness score for the patent
            - novelty_score: novelty score for the patent
            - n_topics: number of topics in the patent
            - topics: list of topics in the patent

    """
    filepath = f"outputs/novelty/patent_novelty_{level}.parquet"
    if from_local:
        return pd.read_parquet(PROJECT_DIR / filepath)
    else:
        return download_obj(
            BUCKET_NAME,
            filepath,
            download_as="dataframe",
        )


def get_patent_topic_pair_commonness(
    level: int = 5, from_local: bool = False
) -> pd.DataFrame:
    """
    Downloads topic pair commonness scores for patents

    Args:
        level (int): Taxonomy level that was used for commonness calculations
        from_local (bool): Whether to load the data from local disk

    Returns:
        pd.DataFrame: Topic pair commonness scores for patents. Columns are:
            - topic_1: first topic in the pair
            - topic_2: second topic in the pair
            - year: priority year
            - N_ij_t: Number of co-occurrences of topic_1 and topic_2 in year t
            - commonness: commonness score for the topic pair in year t

    """
    filepath = f"outputs/novelty/patent_topic_pair_commonness_{level}.parquet"
    if from_local:
        return pd.read_parquet(PROJECT_DIR / filepath)
    else:
        return download_obj(
            BUCKET_NAME,
            filepath,
            download_as="dataframe",
        )


def get_topic_novelty_patents(
    level: int = 5, from_local: bool = False
) -> pd.DataFrame:
    """
    Downloads topic-level novelty score (based on patents).
    There are two novelty scores: one based on aggregating document novelty scores,
    and one based on aggregating topic pair novelty scores.

    Note that for more granular levels, many of the topic pairs novelty scores
    might be missing if they didn't meet the minimum number of pair count threshold.
    To overcome this, one could use different pair count thresholds for different
    levels of the taxonomy.

    Args:
        level (int): Taxonomy level that was used for commonness calculations
        from_local (bool): Whether to load the data from local disk

    Returns:
        pd.DataFrame: Table with topic-level novelty scores. Columns are:
            - topic: Topic identifier
            - topic_name: Human-readable name of the topic
            - year: Year to which the novelty score corresponds to
            - topic_doc_novelty: Novelty measure based on aggregating document novelty scores
            - doc_counts: Number of documents with the topic in the given year
            - topic_pair_novelty: Alternative novelty measure based on aggregating topic pair novelty scores
            - topic_pair_commonness: Aggregated topic commonness measure (ie, the inverse of topic pair novelty)
            - pair_counts: Number of topic pairs with the topic in the given year
    """
    filepath = f"outputs/novelty/topic_novelty_patents_{level}.parquet"
    if from_local:
        return pd.read_parquet(PROJECT_DIR / filepath)
    else:
        return download_obj(
            BUCKET_NAME,
            filepath,
            download_as="dataframe",
        )