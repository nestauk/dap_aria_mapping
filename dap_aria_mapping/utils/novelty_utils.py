"""
Utils for measuring 'novelty' of research publication and patent abstracts

The novelty score has been proposed by Uzzi et al. (2013) and revised by Lee et al. (2015). 
Here we have adapted the method to use topics instead of journal citations.

The score is calculated as follows: 

Step 1: Calculate the commonness of each pair of topics in each year

commonness(i,j) = N(i,j,t) * N(t) / (N(i,t)*N(j,t))

where:
- N(i,j,t) is the number of co-occurrences of i and j in time t (t = usually in a given year)
- N(i, t) is the number of pairs of topics that include topic i
- N(j, t) is the number of pairs of topics that include topic j
- N(t) is the number of pairs of topics in time t

Step 2: Calculate the novelty score for each document

The novelty score at a document level is then calculated by:
- Taking the 10th percentile of its topic pair commonness scores
- Calculating the negative natural log of this value

"""
from dap_aria_mapping import logging
import pandas as pd
import itertools
import numpy as np
from typing import Tuple, Dict


def preprocess_topics_dict(
    topics_dict: Dict[str, list],
    metadata_df: pd.DataFrame,
    id_column: str = "work_id",
    year_column: str = "publication_year",
) -> pd.DataFrame:
    """
    Preprocess a dictionary of topics to a dataframe with columns "work_id", "year" and "topics"

    Args:
        topics_dict (dict): A dictionary with document IDs as keys and a list of topics as values
        metadata_df (pd.DataFrame): Metadata dataframe with columns id_column and publication_year
        id_column (str, optional): Name of the column in metadata_df that contains the document IDs. Defaults to "work_id".
        year_column (str, optional): Name of the column in metadata_df that contains the publication year. Defaults to "publication_year".

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
        id_column (str, optional): Name of the column that contains document ids. Defaults to "work_id".

    Returns:
        pd.DataFrame: A dataframe with document novelty, with columns "work_id", "year", "novelty", "n_topics" and "topics"
        pd.DataFrame: A dataframe with topic pair commonness, with columns "work_id", "topic_1", "topic_2", "year", "commonness"
    """
    # Table where each row corresponds to a document
    # and one pair of topics in that document
    doc_topic_pairs = get_document_topic_pairs(topics_df, id_column=id_column)
    # Number of times each pair of topics appears each years across the data sample
    topic_pair_counts = get_topic_pair_counts(doc_topic_pairs, id_column=id_column)
    # Unique years in the data sample
    years = sorted(topic_pair_counts["year"].unique())

    work_novelty = []
    topic_pair_commonness_list = []

    for year in years:
        logging.info(f"Processing year {year}")
        topic_pair_counts_in_year = topic_pair_counts.query("year == @year")
        # Get all N_i_t / N_j_t values (counts of the pairs that a topic appears in, for a given year)
        topic_counts_dict = get_counts_of_pairs_with_topic(
            topic_pair_counts_in_year, id_column
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
        work_novelty_df = aggregate_document_novelty(topic_pair_commonness, id_column)
        work_novelty.append(work_novelty_df)

        # Save the topic pair commonness values for each year
        topic_pair_commonness_list.append(
            topic_pair_commonness.drop_duplicates(["topic_1", "topic_2", "year"]).drop(
                [id_column, "N_i_t", "N_j_t", "topic_pair"], axis=1
            )
        )

    return (
        pd.concat(work_novelty, ignore_index=True)
        # Add topics labels and number of topics to the document-level novelty dataframe
        .merge(topics_df[[id_column, "n_topics", "topics"]], on=id_column, how="left")
    ), (pd.concat(topic_pair_commonness_list, ignore_index=True))


def get_document_topic_pairs(
    document_table: pd.DataFrame, id_column: str = "work_id"
) -> pd.DataFrame:
    """
    Given a document table with document ids, years and list of topics, create a table with document_ids
    and all pairwise combinations of topics in respective documents

    Args:
        document_table (pd.DataFrame): Dataframe with columns id_column, "years" and "topics"
        id_column (str, optional): Name of the column that contains document ids. Defaults to "document_id".

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
    id_column = "work_id",
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
    ).agg(counts=(id_column, "count"))


def get_counts_of_pairs_with_topic(
    document_topic_pairs: pd.DataFrame, id_column="work_id"
) -> Dict[str, int]:
    """
    Calcuate all N_i_t / N_j_t values:
    Given a table with document ids and topic pairs mentioned in the document,
    calculates the number of pairs in which the topic pair appears

    NB: The input dataframe should contain topic pairs only for a single year

    Args:
        document_topic_pairs (dict): Dataframe with columns for "work_id", "topic_1", "topic_2", "counts"
        id_column (str, optional): Name of the column that contains document ids. Defaults to "work_id".

    Returns:
        Dict[str, int]: Dictionary with topic as key and pair counts as value
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
    topic_counts_dict: Dict[str, int],
    topic_pair_counts_dict: Dict[str, int],
) -> pd.DataFrame:
    """
    Calculate the commonness score for each topic pair in each document

    Args:
        document_topic_pairs (pd.DataFrame): Dataframe with columns "work_id", "topic_1", "topic_2", "year"
        N_t (int): Total number of documents
        topic_counts_dict (dict): Dictionary with topic counts
        topic_pair_counts_dict (dict): Dictionary with topic pair counts

    Returns:
        pd.DataFrame: Dataframe with a commonness score for each topic pair in each document
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
        .assign(novelty=lambda df: convert_commonness_to_novelty(df["commonness"]))
    )


def convert_commonness_to_novelty(commonness_score: float) -> float:
    """
    Calculate novelty score as the negative natural log of commonness score
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


def document_to_topic_novelty(
    document_novelty: pd.DataFrame,
    id_column: str = "work_id",
    aggregation: str = "median",
) -> pd.DataFrame:
    """
    Explodes the topics column and aggregates the novelty scores

    Args:
        document_novelty (pd.DataFrame): A dataframe with novelty scores and list of topics for each document
        id_column (str, optional): The column name for the document id. Defaults to "work_id". Defaults to "work_id".
        aggregation (str, optional): The aggregation to use. Defaults to "median".

    Returns:
        pd.DataFrame: A dataframe with "topic", "year", "doc_counts" and "topic_doc_novelty" novelty score
    """
    return (
        document_novelty.explode("topics")
        .groupby(["topics", "year"], as_index=False)
        .agg(
            topic_doc_novelty=("novelty", aggregation), doc_counts=(id_column, "count")
        )
        .rename(columns={"topics": "topic"})
    )


def pair_to_topic_novelty(
    topic_pair_commonness: pd.DataFrame,
    aggregation: str = "10th_percentile",
    min_counts: int = 0,
) -> pd.DataFrame:
    """
    Calculate the novelty score of a topic by aggregating the commonness scores of all topic pairs in which it appears

    Args:
        topic_pair_commonness (pd.DataFrame): Dataframe with columns "topic_1", "topic_2", "year", "commonness", "N_ij_t"
        aggregation (str, optional): Method that will be used by the pd.DataFrame.groupby().agg() to aggregate commonness scores.
            Defaults to '10th percentile'.
        min_counts (int, optional): Minimum number of times a topic pair must appear to be included in the calculation. Defaults to 0.

    Returns:
        pd.DataFrame: A dataframe with "topic", "year", "topic_pair_novelty" and "pair_counts"
    """
    # Convert topic pair commonness table into a long format,
    # by assigning each topic of a topic pair its own row
    df = pd.concat(
        [
            topic_pair_commonness[["topic_1", "year", "commonness", "N_ij_t"]]
            .query("N_ij_t > @min_counts")
            .rename(columns={"topic_1": "topic"}),
            topic_pair_commonness[["topic_2", "year", "commonness", "N_ij_t"]]
            .query("N_ij_t > @min_counts")
            .rename(columns={"topic_2": "topic"}),
        ],
        ignore_index=True,
    )
    # Aggregate the commonness scores across topics, and convert to a novelty score
    if aggregation == "10th_percentile":
        aggregation = lambda x: aggregate_document_commonness(x, percentile=10)
    else:
        aggregation = aggregation
    return (
        df.groupby(["topic", "year"], as_index=False)
        .agg(
            topic_pair_commonness=("commonness", aggregation),
            pair_counts=("N_ij_t", "sum"),
        )
        .assign(
            topic_pair_novelty=lambda df: df.topic_pair_commonness.apply(
                convert_commonness_to_novelty
            )
        )
    )

def create_topic_to_document_dict(
    document_novelty_df,
    document_df,
    id_column="work_id",
    document_title_column="display_name"
):
    """Generates dictionary for faster lookups"""
    return (
            document_novelty_df[[id_column, "topics", "novelty", "year", "n_topics"]]
            .merge(document_df[[id_column, document_title_column]], on=id_column, how="left")
            .groupby("topics")
            .agg(lambda x: x.tolist())
            .to_dict(orient='index')
        )


