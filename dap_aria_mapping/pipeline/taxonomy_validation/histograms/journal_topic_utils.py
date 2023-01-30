import pandas as pd
import altair as alt
from nesta_ds_utils.viz.altair import formatting, saving

formatting.setup_theme()
from toolz import pipe
from typing import Union
from functools import partial

from dap_aria_mapping.getters.taxonomies import get_topic_names
from dap_aria_mapping.utils.histograms import (
    get_relevant_topics,
    generate_sample,
    clean_topic_ids,
    sort_topics,
)
from dap_aria_mapping.utils.topic_names import *


def build_unary_data(
    df: pd.DataFrame,
    taxonomy_class: str,
    level: int = 1,
    threshold: float = 0.8,
    sample: Union[int, str] = "main",
    **kwargs,
) -> pd.DataFrame:
    """Build a dataframe of unary data for a given taxonomy class and level.
        The dataframe contains the following columns:
            - Journal: The journal name
            - Entity: The entity name
            - Level: The level of the taxonomy
            - Topic: The topic name
            - Count: The number of times the entity in a journal was assigned
                to the topic.

    Args:
        df (pd.DataFrame): A dataframe of Entity x Journal x Level assignments.
        taxonomy_class (str): A taxonomy class to build unary data for.
        level (int, optional): The level of the taxonomy to build unary data for.
        threshold (float, optional): The threshold for the percentage of total assignments
        sample (Union[int, str], optional): The number of journals to sample from the
            dataframe. If "main", will sample from the top 50 journals in the dataframe.
        **kwargs: Additional keyword arguments to pass to the get_relevant_topics.
            The only additional argument is parent_topic, which is a list of parent
            topics to filter the dataframe by.

    Returns:
        pd.DataFrame: A dataframe of unary data for a given taxonomy class and level.
    """

    SAMPLE_JOURNALS = generate_sample(df, level, sample, **kwargs)

    journal_counts = {
        journal: get_relevant_topics(
            df=df,
            level=level,
            journal=journal,
            threshold=threshold,
            method="relative",
            **kwargs,
        )
        for journal in SAMPLE_JOURNALS
    }

    if level == 1:
        n_entities = {k: df["Count"].sum() for k in df["Level_1"].unique()}
    elif level > 1:
        n_entities = {
            k: get_n_entities(df, level, **kwargs)
            for k in df[f"Level_{str(level)}"].unique()
        }

    journal_unary_df = pipe(
        pd.DataFrame(
            [
                (k, val, n_entities[val])
                for k, v in journal_counts.items()
                for val in v.keys()
            ],
            columns=["Journal", "Topic", "Total_Count"],
        ),
        lambda df: (
            df.assign(freq=df.groupby("Journal")["Topic"].transform("count"))
            .sort_values(["freq", "Journal"], ascending=False)
            .drop("freq", axis=1)
        ),
    )

    dict_entities = get_topic_names(taxonomy_class, "entity", level)

    journal_unary_df["Entities"] = journal_unary_df["Topic"].map(dict_entities)

    journal_unary_df["Topic"] = clean_topic_ids(journal_unary_df)

    return journal_unary_df


def make_unary_histogram(
    df: pd.DataFrame,
    taxonomy_class: str = None,
    level: int = 1,
    save: bool = False,
    **kwargs,
) -> alt.Chart:
    """Make a histogram of journals and their relevant associated topics.

    Args:
        df (pd.DataFrame): A dataframe of Entity x Journal x Level assignments.
        taxonomy_class (str, optional): The taxonomy class to use. Defaults to None.
        level (int, optional): The level of the taxonomy to return values for. Defaults to 1.

    Returns:
        alt.Chart: A histogram of journals and their relevant associated topics.
    """
    if len(df) == 0:
        return None

    entity_count = str(df["Total_Count"].values[0]) if len(df) > 0 else "0"

    if level == 1:
        title = f"Taxonomy: {taxonomy_class} - Count = {entity_count}"
    else:
        assert (
            "parent_topic" in kwargs.keys()
        ), "Must provide parent_topic argument for levels > 1"
        names = get_topic_names(taxonomy_class, "entity", level - 1)
        title = f"Taxonomy: {taxonomy_class} - Parent Topic {kwargs['parent_topic']}: {names[kwargs['parent_topic']]} - Count = {entity_count}"

    chart = (
        alt.Chart(df, title=title)
        .mark_bar()
        .encode(
            x=alt.X(
                "Journal:N", axis=alt.Axis(labelAngle=90, labelLimit=200), sort=None
            ),
            y=alt.Y("count(Topic):Q", axis=alt.Axis(tickMinStep=1)),
            tooltip=["Journal", "Entities"],
            color=alt.Color("Topic:N", sort="ascending"),
        )
        .properties(width=1600, height=600)
        .interactive()
    )

    if save:
        chart.save(
            PROJECT_DIR
            / "outputs"
            / "figures"
            / "taxonomy_validation"
            / f"{taxonomy_class}_histogram_level_{level}_partopic_{kwargs['parent_topic']}.html"
        )
    else:
        return chart


def make_unary_binary_chart(
    df: pd.DataFrame,
    taxonomy_class: str = None,
    level: int = 1,
    save: bool = False,
    **kwargs,
) -> alt.Chart:
    """Make a binary histogram of journals and their relevant associated topics.

    Args:
        df (pd.DataFrame): A dataframe of Entity x Journal x Level assignments.
        taxonomy_class (str, optional): The taxonomy class to use. Defaults to None.
        level (int, optional): The level of the taxonomy to return values for. Defaults to 1.

    Returns:
        alt.Chart: A histogram of journals and their relevant associated topics.
    """
    if len(df) == 0:
        return None

    entity_count = str(df["Total_Count"].values[0]) if len(df) > 0 else "0"

    if level == 1:
        title = f"Taxonomy: {taxonomy_class} - Count = {entity_count}"
    else:
        assert (
            "parent_topic" in kwargs.keys()
        ), "Must provide parent_topic argument for levels > 1"
        names = get_topic_names(taxonomy_class, "entity", level - 1)
        title = f"Taxonomy: {taxonomy_class} - Parent Topic {kwargs['parent_topic']}: {names[kwargs['parent_topic']]} - Count = {entity_count}"

    chart = (
        alt.Chart(df, title=title)
        .mark_bar()
        .encode(
            x=alt.X(
                "Journal:N", axis=alt.Axis(labelAngle=90, labelLimit=200), sort=None
            ),
            y=alt.Y("Topic:N", sort=sort_topics(df)),
            tooltip=["Journal", "Entities"],
            color=alt.Color("Topic:N", sort="ascending"),
        )
        .properties(width=1600, height=800)
        .interactive()
    )

    if save:
        chart.save(
            PROJECT_DIR
            / "outputs"
            / "figures"
            / "taxonomy_validation"
            / f"{taxonomy_class}_heatmap_level_{level}_partopic_{kwargs['parent_topic']}.html"
        )
    else:
        return chart


def build_frequency_data(
    df: pd.DataFrame,
    taxonomy_class: str,
    level: int = 1,
    threshold: float = 1.0,
    sample: Union[int, str] = "main",
    **kwargs,
) -> pd.DataFrame:
    """Build a dataframe of frequency data for a given taxonomy class and level.
        The dataframe contains the following columns:
            - Journal: The journal name
            - Entity: The entity name
            - Level: The level of the taxonomy
            - Topic: The topic name
            - Value: The frequency with which a topic is present in a journal.

    Args:
        df (pd.DataFrame): A dataframe of Entity x Journal x Level assignments.
        taxonomy_class (str): A taxonomy class to build unary data for.
        level (int, optional): The level of the taxonomy to build unary data for.
        threshold (float, optional): The threshold for the percentage of total assignments
        sample (Union[int, str], optional): The number of journals to sample from the
            dataframe. If "main", will sample from the top 50 journals in the dataframe.
        **kwargs: Additional keyword arguments to pass to the get_relevant_topics.
            The only additional argument is parent_topic, which is a list of parent
            topics to filter the dataframe by.

    Returns:
        pd.DataFrame: A dataframe of frequency data for a given taxonomy class and level.
    """

    SAMPLE_JOURNALS = generate_sample(df, level, sample, **kwargs)

    journal_counts = {
        journal: get_relevant_topics(
            df=df,
            level=level,
            journal=journal,
            threshold=threshold,
            method="relative",
            **kwargs,
        )
        for journal in SAMPLE_JOURNALS
    }

    if level == 1:
        n_entities = {k: df["Count"].sum() for k in df["Level_1"].unique()}
    elif level > 1:
        n_entities = {
            k: get_n_entities(df, level, **kwargs)
            for k in df[f"Level_{str(level)}"].unique()
        }

    journal_freq_df = pd.DataFrame(
        [
            (k, key, val, n_entities[key])
            for k, v in journal_counts.items()
            for key, val in v.items()
        ],
        columns=["Journal", "Topic", "Value", "Total_Count"],
    )

    dict_entities = get_topic_names(taxonomy_class, "entity", level)

    journal_freq_df["Entities"] = journal_freq_df["Topic"].map(dict_entities)

    journal_freq_df["Topic"] = clean_topic_ids(journal_freq_df)

    return journal_freq_df


def make_freq_barplot(
    df: pd.DataFrame,
    taxonomy_class: str = None,
    level: int = 1,
    save: bool = False,
    **kwargs,
) -> alt.Chart:
    """Make a stacked barplot of journals and their relevant associated topics.

    Args:
        df (pd.DataFrame): A dataframe of Entity x Journal x Level assignments.
        taxonomy_class (str, optional): The taxonomy class to use. Defaults to None.
        level (int, optional): The level of the taxonomy to return values for. Defaults to 1.

    Returns:
        alt.Chart: A barplot of journals and all their associated topics.
    """
    if len(df) == 0:
        return None
    entity_count = str(df["Total_Count"].values[0]) if len(df) > 0 else "0"

    if level == 1:
        title = f"Taxonomy: {taxonomy_class} - Count = {entity_count}"
    else:
        assert (
            "parent_topic" in kwargs.keys()
        ), "Must provide parent_topic argument for levels > 1"
        names = get_topic_names(taxonomy_class, "entity", level - 1)
        title = f"Taxonomy: {taxonomy_class} - Parent Topic {kwargs['parent_topic']}: {names[kwargs['parent_topic']]} - Count = {entity_count}"

    chart = (
        alt.Chart(df, title=title)
        .mark_bar()
        .encode(
            x=alt.X(
                "Journal:N",
                axis=alt.Axis(labelAngle=90, labelLimit=200),
                sort=list(
                    df.groupby(["Journal"], sort=False)["Value"]
                    .max()
                    .sort_values()
                    .index
                ),
            ),
            y=alt.Y("Value:Q", axis=alt.Axis(tickMinStep=0.2), sort=sort_topics(df)),
            tooltip=["Journal", "Entities"],
            color=alt.Color("Topic:N", sort="ascending"),
        )
        .properties(width=1600, height=600)
        .interactive()
    )

    if save:
        chart.save(
            PROJECT_DIR
            / "outputs"
            / "figures"
            / "taxonomy_validation"
            / f"{taxonomy_class}_frequency_level_{level}_partopic_{kwargs['parent_topic']}.html"
        )

    else:
        return chart


def get_n_entities(df: pd.DataFrame, level: int, **kwargs) -> int:
    """Get the number of entities at a given level.

    Args:
        df (pd.DataFrame): A dataframe of Entity x Journal x Level assignments.
        level (int): The level of the taxonomy to return values for.

    Returns:
        int: The number of entities at a given level.
    """
    df = df.loc[df[f"Level_{str(level - 1)}"].isin([kwargs["parent_topic"]])]
    return df["Count"].sum()


def build_tfidf(np_array: np.ndarray) -> np.ndarray:
    """Builds a J x T matrix, where J corresponds to the number
        of unique journals and T the number of unique topics.

    Args:
        np_array (np.ndarray): J x T array of journal x topic counts.

    Returns:
        np.ndarray: TF-IDF weights for all journal x topic pairs.
    """
    tf = np_array / np_array.sum(axis=1)[:, None]
    idf = np.log(np_array.shape[0] / (np.count_nonzero(np_array, axis=0)))
    return tf * idf


def get_max(series: pd.Series) -> float:
    """Gets the maximum value of a normalised series of values.

    Args:
        series (pd.Series): A series of topic-journal values.

    Returns:
        float: The maximum value of a topic-journal pair, normalised
            by the sum of TF-IDF values of the pair.
    """
    transformed_vals = 100 * (series / series.sum())
    return transformed_vals.max()


def build_tfidf_data(
    df: pd.DataFrame,
    taxonomy_class: str,
    level: int = 1,
    threshold: float = 1.0,
    sample: Union[int, str] = "main",
    **kwargs,
) -> pd.DataFrame:
    """Build a dataframe of TF-IDF outputs for a given taxonomy class and level,
        where documents correspond to journals and tokens to topics.The dataframe
        contains the following columns:
            - Journal: The journal name
            - Entity: The entity name
            - Level: The level of the taxonomy
            - Topic: The topic name
            - Value: The TF-IDF value for a topic-journal pair.

    Args:
        df (pd.DataFrame): A dataframe of Entity x Journal x Level assignments.
        taxonomy_class (str): A taxonomy class to build unary data for.
        level (int, optional): The level of the taxonomy to build unary data for.
        threshold (float, optional): The threshold for the percentage of total assignments
        sample (Union[int, str], optional): The number of journals to sample from the
            dataframe. If "main", will sample from the top 50 journals in the dataframe.
        **kwargs: Additional keyword arguments to pass to the get_relevant_topics.
            The only additional argument is parent_topic, which is a list of parent
            topics to filter the dataframe by.

    Returns:
        pd.DataFrame: A dataframe of TF-IDF data for a given taxonomy class and level.
    """

    journal_counts = {
        journal: get_relevant_topics(
            df=df,
            level=level,
            journal=journal,
            threshold=threshold,
            method="absolute",
            **kwargs,
        )
        for journal in df["Journal"].unique()
    }

    if level == 1:
        n_entities = {k: df["Count"].sum() for k in df["Level_1"].unique()}
    elif level > 1:
        n_entities = {
            k: get_n_entities(df, level, **kwargs)
            for k in (
                df.loc[df[f"Level_{str(level - 1)}"].isin([kwargs["parent_topic"]])][
                    f"Level_{str(level)}"
                ].unique()
            )
        }

    journal_counts = pipe(
        pd.DataFrame(
            [
                (k, key, val)
                for k, v in journal_counts.items()
                for key, val in v.items()
            ],
            columns=["Journal", "Topic", "Value"],
        ),
        partial(
            pd.pivot_table,
            index="Journal",
            columns="Topic",
            values="Value",
            aggfunc="sum",
        ),
    )

    SAMPLE_JOURNALS = pipe(
        generate_sample(df, level, sample, **kwargs),
        lambda ls: [j for j in journal_counts.index if j in ls],
    )

    journals_tfidf = pipe(
        journal_counts,
        lambda df: df.fillna(0).to_numpy(),
        build_tfidf,
        partial(
            pd.DataFrame, columns=journal_counts.columns, index=journal_counts.index
        ),
        lambda df: df.loc[SAMPLE_JOURNALS],
        lambda df: pd.melt(df.reset_index(), id_vars="Journal"),
        lambda df: df.loc[df.value > 0.0],
    )

    dict_entities = get_topic_names(taxonomy_class, "entity", level)
    journals_tfidf["Entities"] = journals_tfidf["Topic"].map(dict_entities)
    journals_tfidf["Total_Count"] = journals_tfidf["Topic"].map(n_entities)
    journals_tfidf["Topic"] = clean_topic_ids(journals_tfidf)

    return journals_tfidf


def make_tfidf_barplot(
    df: pd.DataFrame,
    taxonomy_class: str = None,
    level: int = 1,
    save: bool = False,
    **kwargs,
) -> alt.Chart:
    """Make a stacked barplot of journals and their relevant associated topics.

    Args:
        df (pd.DataFrame): A dataframe of Entity x Journal x Level assignments.
        taxonomy_class (str, optional): The taxonomy class to use. Defaults to None.
        level (int, optional): The level of the taxonomy to return values for. Defaults to 1.

    Returns:
        alt.Chart: A barplot of journals and all their associated topics.
    """
    if len(df) == 0:
        return None
    entity_count = str(df["Total_Count"].values[0]) if len(df) > 0 else "0"

    if level == 1:
        title = f"Taxonomy: {taxonomy_class} - Count = {entity_count}"
    else:
        assert (
            "parent_topic" in kwargs.keys()
        ), "Must provide parent_topic argument for levels > 1"
        names = get_topic_names(taxonomy_class, "entity", level - 1)
        title = f"Taxonomy: {taxonomy_class} - Parent Topic {kwargs['parent_topic']}: {names[kwargs['parent_topic']]} - Count = {entity_count}"

    chart = (
        alt.Chart(df, title=title)
        .mark_bar()
        .encode(
            x=alt.X(
                "Journal:N",
                axis=alt.Axis(labelAngle=90, labelLimit=200),
                sort=list(
                    df.groupby("Journal")["value"]
                    .agg(lambda group: get_max(group))
                    .sort_values()
                    .index
                ),
            ),
            y=alt.Y("value:Q", axis=alt.Axis(tickMinStep=0.2)),
            tooltip=["Journal", "Entities"],
            color=alt.Color("Topic:N", sort="ascending"),
        )
        .properties(width=1600, height=600)
        .interactive()
    )

    if save:
        chart.save(
            PROJECT_DIR
            / "outputs"
            / "figures"
            / "taxonomy_validation"
            / f"{taxonomy_class}_tfidf_level_{level}_partopic_{kwargs['parent_topic']}.html"
        )
    else:
        return chart
