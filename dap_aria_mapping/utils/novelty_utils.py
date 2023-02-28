"""
Utils for measuring 'novelty' of research publication and patent abstracts
"""
import pandas as pd
import itertools
import numpy as np

def commoness_score(df: pd.DataFrame, year: int, topic_i: str, topic_j: str) -> float:
    """
    Calculate the commonness score for a given topic pair and year

    commonness(i,j) = N(i,j,t) * N(t) / (N(i,t)*N(j,t))

    where:
    - t is the year
    - N(i,j,t) is the number of co-occurrences of i and j in year t,
    - N(i,t) is the number of pairs of topics that include topic i in year t
    - N(j,t) is the number of pairs of topics that include topic j in year t
    - N(t) is the number of pairs of topics in year t

    Args:
        df (pd.DataFrame): Dataframe with columns "topic_1", "topic_2", "year" and "counts"
        year (int): Year to calculate the novelty score for
        topic1 (str): First topic of the pair
        topic2 (str): Second topic of the pair

    Returns:
        float: The commonness score
    """
    # Calculate N(i,j,t)
    N_ij_t = df[
        (df["topic_1"] == topic_i) & (df["topic_2"] == topic_j) & (df["year"] == year)
    ]["counts"].sum()
    # Calculate N(i,t)
    N_i_t = df[
        ((df["topic_1"] == topic_i) | (df["topic_2"] == topic_i)) & (df["year"] == year)
    ]["counts"].sum()
    # Calculate N(j,t)
    N_j_t = df[
        ((df["topic_1"] == topic_j) | (df["topic_2"] == topic_j)) & (df["year"] == year)
    ]["counts"].sum()
    # Calculate N(t)
    N_t = df[df["year"] == year]["counts"].sum()
    # Calculate commonness
    return N_ij_t * N_t / (N_i_t * N_j_t)


def novelty_score(commonness_score: float) -> float:
    """
    Calculate novelty as the negative natural log of commonness score
    """
    return -np.log(commonness_score)


def document_commonness(
    document_df: pd.DataFrame, aggregation_method: str = "mean", id_column="document_id"
) -> pd.DataFrame:
    """
    Calculate commonness of a document by aggregating the commonness of all topic pairs 
    mentioned in the document

    Args:
        document_df (pd.DataFrame): Dataframe with columns "document_id", "years" and "topics"
        aggregation_method (str, optional): Aggregation method to use. Defaults to 'mean'.

    Returns:
        pd.DataFrame: A dataframe with columns "document_id", "year", "commonness"
    """
    # Calculate the commonness score for each topic pair and year
    df = topic_pair_commonness(document_df, id_column=id_column)
    # Aggregate the commonness score for each document and year
    df = (
        df.groupby([id_column, "year"])
        .agg({"commonness": aggregation_method})
        .reset_index()
    )
    # Return the dataframe
    return df


def document_novelty(
        document_df: pd.DataFrame, aggregation_method: str = "mean", id_column="document_id"
) -> pd.DataFrame:
    """
    Calculate novelty of a document

    Args:
        document_df (pd.DataFrame): Dataframe with columns "document_id", "years" and "topics"
        aggregation_method (str, optional): Aggregation method to use. Defaults to 'mean'.
        id_column (str, optional): Name of the column with document ids. Defaults to "document_id".

    Returns:
        pd.DataFrame: A dataframe with columns "document_id", "year", "novelty"
    """
    return (
        document_commonness(document_df, aggregation_method, id_column)
        .assign(novelty=lambda x: x["commonness"].apply(novelty_score))
        .drop("commonness", axis=1)
    )


def topic_pair_commonness(
    document_df: pd.DataFrame, id_column: str = "document_id"
) -> pd.DataFrame:
    """
    Given a document table with document ids, years and list of topics, calculate the commonness score
    for each topic pair and year

    Args:
        document_df (pd.DataFrame): Dataframe with columns "document_id", "years" and "topics"

    Returns:
        pd.DataFrame: A dataframe with columns "topic_1", "topic_2", "year", "commonness"
    """
    # Create a table with document ids, year and topic pairs mentioned in the document
    document_topic_pairs = get_document_topic_pairs(document_df, id_column=id_column)
    # Calculate the commonness score for each topic pair and year
    return document_topic_pairs.merge(
        yearly_commonness_of_topic_pairs(document_topic_pairs),
        on=["topic_1", "topic_2", "year"],
        how="left",
    )


def yearly_commonness_of_topic_pairs(
    document_topic_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Given a table with document ids, year and topic pairs mentioned in the document,
    calculate the commonness score for each topic pair and year

    Args:
        document_topic_pairs (pd.DataFrame): Dataframe with columns "document_id", "topic_1", "topic_2", "year"

    Returns:
        pd.DataFrame: A dataframe with columns "topic_1", "topic_2", "year", "commonness"
    """
    # Calculate the number of co-occurrences of each topic pair in each year
    df = (
        document_topic_pairs.groupby(["topic_1", "topic_2", "year"])
        .size()
        .reset_index(name="counts")
    )
    # Calculate the commonness score for each topic pair and year
    df["commonness"] = df.apply(
        lambda x: commoness_score(df, x["year"], x["topic_1"], x["topic_2"]), axis=1
    )
    # Return the dataframe
    return df


def get_document_topic_pairs(
    document_table: pd.DataFrame, id_column: str = "document_id"
) -> pd.DataFrame:
    """
    Given a document table with document ids, years and list of topics, create a table with document_ids
    and all pairwise combinations of topics in respective documents

    Args:
        document_table (pd.DataFrame): Dataframe with columns id_column, "years" and "topics"
        id_column (str, optional): Name of the column that contains document ids. Defaults to 'document_id'.

    Returns:
        pd.DataFrame: A dataframe with columns "document_id", "topic_1", "topic_2", "year"
    """
    # Create a list of all document - topic pairs
    document_topic_pairs = []
    for _, row in document_table.iterrows():
        document_id = row[id_column]
        year = row["year"]
        # Deduplicate topics
        topics = list(set(row["topics"]))
        for topic1, topic2 in itertools.combinations(topics, 2):
            # Make sure that the topics are sorted alphabetically/numerically
            if topic1 > topic2:
                topic1, topic2 = topic2, topic1
            document_topic_pairs.append([document_id, topic1, topic2, year])
    # Return a dataframe            
    return pd.DataFrame(
        data=document_topic_pairs, columns=[id_column, "topic_1", "topic_2", "year"]
    )
