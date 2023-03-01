"""
Utils for measuring 'novelty' of research publication and patent abstracts
"""
from dap_aria_mapping import logging
import pandas as pd
import itertools
import numpy as np
from typing import Tuple


def preprocess_topics_dict(
    topics_dict: dict,
    metadata_df: pd.DataFrame,
    id_column: str = "work_id",
    year_column: str = "publication_year",
) -> pd.DataFrame:
    """
    Preprocess a dictionary of topics to a dataframe with columns "work_id", "year" and "topics"

    Args:
        topics_dict (dict): A dictionary with document IDs as keys and a list of topics as values
        metadata_df (pd.DataFrame): Metadata dataframe with columns id_column and publication_year
        id_column (str): Name of the column in metadata_df that contains the document IDs
        year_column (str): Name of the column in metadata_df that contains the publication year

    Returns:
        pd.DataFrame: A dataframe with columns "work_id", "year" and "topics"
    """
    return (
        # Convert to dataframe
        pd.DataFrame(
            data={
                id_column: list(topics_dict.keys()),
                "topics": list(topics_dict.values()),
            }
        )
        # Deduplicate the topics in the list in 'topics' column in each row
        .assign(topics=lambda df: df["topics"].apply(lambda x: list(set(x))))
        # Drop rows with less than two topics (as we need at least two topics for a combination to exist)
        .assign(n_topics=lambda df: df["topics"].apply(lambda x: len(x)))
        .query("n_topics >= 2")
        # Merge with metadata to get publication year
        .merge(metadata_df[[id_column, year_column]], on=id_column, how="left")
        .rename(columns={year_column: "year"})
        # Drop rows with missing year, as can't be used for novelty calculation
        .dropna(subset=["year"])
        .astype({"year": "int"})
    )


def document_novelty(
    topics_df: pd.DataFrame,
    id_column: str = "work_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate novelty scores for each document in a dataframe of topics

    Args:
        topics_df (pd.DataFrame): A dataframe with columns for document id, "year" and "topics"
        id_column (str, optional): Name of the column that contains document ids. Defaults to 'work_id'.

    Returns:
        pd.DataFrame: A dataframe with document novelty, with columns "work_id", "year", "novelty", "n_topics" and "topics"
        pd.DataFrame: A dataframe with topic pair commonness, with columns "work_id", "topic_1", "topic_2", "year", "commonness"
    """
    # Table where each row corresponds to a document
    # and one pair of topics in that document
    doc_topic_pairs = get_document_topic_pairs(topics_df, id_column=id_column)
    # Number of times each pair of topics appears each years across the data sample
    topic_pair_counts = get_topic_pair_counts(doc_topic_pairs)
    # Unique years in the data sample
    years = sorted(topic_pair_counts["year"].unique())

    work_novelty = []
    topic_pair_commonness_list = []

    for year in years:
        logging.info(f"Processing year {year}")
        topic_pair_counts_in_year = topic_pair_counts.query("year == @year")
        # Get all N_i_t / N_j_t values (counts of the pairs that a topic appears in, for a given year)
        topic_counts_dict = get_counts_of_pairs_with_topic(
            topic_pair_counts_in_year, "work_id"
        )
        # Get N_t (all topic pair counts)
        N_t = topic_pair_counts_in_year["counts"].sum()
        # Get all N_i_j_t values (topic pair counts)
        topic_pair_counts_dict = dict(
            zip(
                topic_pair_counts_in_year["topic_pair"],
                topic_pair_counts_in_year["counts"],
            )
        )

        # Calculate commonness values for each topic pair
        topic_pair_commonness = get_topic_pair_commonness(
            doc_topic_pairs.query("year == @year"),
            N_t,
            topic_counts_dict,
            topic_pair_counts_dict,
        )

        # Aggregate commonness values at the level of documents and convert to novelty
        work_novelty_df = aggregate_document_novelty(topic_pair_commonness, "work_id")
        work_novelty.append(work_novelty_df)

        # Save the topic pair commonness values for each year
        topic_pair_commonness_list.append(
            topic_pair_commonness.drop_duplicates(["topic_1", "topic_2", "year"]).drop(
                ["N_i_t", "N_j_t", "topic_pair"], axis=1
            )
        )

    return (
        pd.concat(work_novelty, ignore_index=True)
        # Add topics labels and number of topics to the document-level novelty dataframe
        .merge(topics_df[[id_column, "n_topics", "topics"]], on="work_id", how="left")
    ), (pd.concat(topic_pair_commonness_list, ignore_index=True))


def get_document_topic_pairs(
    document_table: pd.DataFrame, id_column: str = "work_id"
) -> pd.DataFrame:
    """
    Given a document table with document ids, years and list of topics, create a table with document_ids
    and all pairwise combinations of topics in respective documents

    Args:
        document_table (pd.DataFrame): Dataframe with columns id_column, "years" and "topics"
        id_column (str, optional): Name of the column that contains document ids. Defaults to 'document_id'.

    Returns:
        pd.DataFrame: A dataframe with columns "work_id", "topic_1", "topic_2", "year"
    """
    # Create a list of all document - topic pairs
    document_topic_pairs = []
    for _, row in document_table.iterrows():
        document_id = row[id_column]
        year = row["year"]
        # Deduplicate topics
        topics = list(set(row["topics"]))
        for topic1, topic2 in itertools.combinations(topics, 2):
            # Make sure that the topics are sorted alphabetically/numerically
            if topic1 > topic2:
                topic1, topic2 = topic2, topic1
            document_topic_pairs.append([document_id, topic1, topic2, year])
    # Return a dataframe
    return pd.DataFrame(
        data=document_topic_pairs, columns=[id_column, "topic_1", "topic_2", "year"]
    ).assign(topic_pair=lambda df: df["topic_1"] + "_x_" + df["topic_2"])


def get_topic_pair_counts(
    document_topic_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Given a table with document ids, year and topic pairs mentioned in the document,
    calculate the number of co-occurrences of each topic pair in each year

    Args:
        document_topic_pairs (pd.DataFrame): Dataframe with columns for "topic_1", "topic_2", "year"

    Returns:
        pd.DataFrame: A dataframe with columns "topic_1", "topic_2", "year", "counts"
    """
    return document_topic_pairs.groupby(
        ["topic_1", "topic_2", "topic_pair", "year"], as_index=False
    ).agg(counts=("work_id", "count"))


def get_counts_of_pairs_with_topic(
    document_topic_pairs: pd.DataFrame, id_column="work_id"
) -> dict:
    """
    Calcuate all N_i_t / N_j_t values:
    Given a table with document ids and topic pairs mentioned in the document,
    calculates the number of pairs in which the topic pair appears

    NB: The input dataframe should contain topic pairs only for a single year

    Args:
        document_topic_pairs (pd.DataFrame): Dataframe with columns "work_id", "topic_1", "topic_2", "year"
    """
    topic_counts_df = (
        pd.concat(
            [
                document_topic_pairs[["topic_1", "counts"]],
                document_topic_pairs[["topic_2", "counts"]],
            ]
        )
        # Combine topic_1 and topic_2 into one column
        .assign(topics=lambda df: df["topic_1"].fillna(df["topic_2"]))
        # Count number of times each topic appears
        .groupby("topics", as_index=False).agg(counts=("counts", "sum"))
    )
    # Convert to dict
    return dict(zip(topic_counts_df["topics"], topic_counts_df["counts"]))


def get_topic_pair_commonness(
    document_topic_pairs: pd.DataFrame,
    N_t: int,
    topic_counts_dict: dict,
    topic_pair_counts_dict: dict,
):
    """
    Calculate the novelty score for each document

    Args:
        document_topic_pairs (pd.DataFrame): Dataframe with columns "work_id", "topic_1", "topic_2", "year"
        N_t (int): Total number of documents
        topic_counts_dict (dict): Dictionary with topic counts
        topic_pair_counts_dict (dict): Dictionary with topic pair counts
    """
    # Calculate the novelty score for each document
    return (
        document_topic_pairs.assign(
            N_ij_t=lambda df: df["topic_pair"].apply(
                lambda x: topic_pair_counts_dict[x]
            )
        )
        .assign(N_i_t=lambda df: df["topic_1"].apply(lambda x: topic_counts_dict[x]))
        .assign(N_j_t=lambda df: df["topic_2"].apply(lambda x: topic_counts_dict[x]))
        .assign(commonness=lambda df: df["N_ij_t"] * N_t / (df["N_i_t"] * df["N_j_t"]))
    )


def aggregate_document_novelty(
    topic_pair_commonness: pd.DataFrame,
    id_column: str = "work_id",
) -> pd.DataFrame:
    """
    Calculate the novelty score for each document
    """
    return (
        topic_pair_commonness.groupby([id_column, "year"], as_index=False)
        .agg(commonness=("commonness", (lambda x: aggregate_document_commonness(x))))
        .assign(novelty=lambda df: novelty_score(df["commonness"]))
    )


def novelty_score(commonness_score: float) -> float:
    """
    Calculate novelty as the negative natural log of commonness score
    """
    return -np.log(commonness_score)


def aggregate_document_commonness(
    commonness_scores: list, percentile: float = 10
) -> float:
    """
    Aggregate commonness scores for a document by taking a specific percentile
    (default: 10th percentile as in the original paper)
    """
    return np.percentile(commonness_scores, percentile)
