import pandas as pd
import random
from typing import Union, Dict, List


def get_relevant_topics(
    df: pd.DataFrame, level: int, journal: str, threshold: float, method: str, **kwargs
) -> Dict[str, Union[int, float]]:
    """Get the counts of relevant topics for a given journal. Relevant topics
        are defined as those that make up a certain percentage of the total
        number of assignments for a given journal. The percentage is defined
        by the threshold argument, and the method argument defines whether
        the function returns the counts of the relevant topics or the
        percentage of the total number of assignments that they make up.

    Args:
        df (pd.DataFrame): A dataframe of Entity x Journal x Level assignments.
        level (int): The level of the taxonomy to return values for.
        journal (str): The journal to return values for.
        threshold (float): The threshold for the percentage of total assignments
            that a topic must make up to be considered relevant. Must be between
            0 and 1.
        method (str): The method to use for calculating the relevant topics.
            Must be either "absolute" or "relative".

    Returns:
        Dict[str, Union[int, float]]: A dictionary of relevant topics and their
            counts or percentages.
    """

    assert 0 < threshold <= 1, "Threshold must be between 0 and 1"
    df_journal = df.loc[df["Journal"] == journal]
    if level > 1:
        df_journal = df_journal.loc[
            df_journal[f"Level_{str(level - 1)}"].isin([kwargs["parent_topic"]])
        ]
    total_assignments = df_journal.shape[0]
    level_assignments = df_journal[f"Level_{str(level)}"].value_counts().to_dict()

    count_assignments = 0
    relevant_topics = {}
    for k, v in level_assignments.items():
        if count_assignments <= threshold * total_assignments:
            if method == "absolute":
                relevant_topics[k] = v
            elif method == "relative":
                relevant_topics[k] = v / total_assignments
            count_assignments += v
        else:
            break
    return relevant_topics


def generate_sample(
    df: pd.DataFrame, level: int, sample: Union[list, int], **kwargs
) -> List[str]:
    """Generate a sample of journals to use for the taxonomy validation.

    Args:
        df (pd.DataFrame): A dataframe of Entity x Journal x Level assignments.
        level (int): The level of the taxonomy to return values for.
        sample (Union[str, int]): The sample to use. If "main", the sample will
            be the top 50 journals in the dataset. If an integer, the sample
            will be a random sample of that size. If an integer and level > 1,
            the sample will be a random sample of that size from the journals
            that are assigned to the parent topics.

    Returns:
        List[str]: A list of journals to use for the taxonomy validation.
    """
    if isinstance(sample, list):
        SAMPLE_JOURNALS = sample
    elif all([isinstance(sample, int), level == 1]):
        SAMPLE_JOURNALS = random.sample(list(df["Journal"].unique()), sample)
    elif all([isinstance(sample, int), level > 1]):
        assert (
            "parent_topic" in kwargs.keys()
        ), "Must provide parent_topic argument for levels > 1"
        SAMPLE_JOURNALS = random.sample(
            list(
                df.loc[df[f"Level_{str(level - 1)}"].isin([kwargs["parent_topic"]])][
                    "Journal"
                ].unique()
            ),
            sample,
        )
    return SAMPLE_JOURNALS


def clean_topic_ids(df: pd.DataFrame, level: int) -> List[str]:
    """Clean the topic ids for a given level of the taxonomy.

    Args:
        df (pd.DataFrame): A dataframe of Entity x Journal x Level assignments.
        level (int): The level of the taxonomy to return values for.

    Returns:
        List[str]: A list of topic ids for the given level.
    """

    topic_ids = df["Topic"].to_list()
    topic_ids = [str(x).split("_")[-1] for x in df["Topic"].to_list()]
    return [int(x) for x in topic_ids]


def sort_topics(df: pd.DataFrame) -> List[str]:
    """Sort topics by their integer value.

    Args:
        df (pd.DataFrame): A dataframe of Entity x Journal x Level assignments.

    Returns:
        List[str]: A list of sorted topics.
    """
    return [x for x in sorted([int(x) for x in df.Topic.unique()])]
