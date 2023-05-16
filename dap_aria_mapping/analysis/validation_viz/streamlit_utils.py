import streamlit as st
import pandas as pd
import numpy as np
import boto3
from collections import defaultdict
from typing import Dict, Sequence, Union
from dap_aria_mapping.getters.simulations import (
    get_simulation_topic_sizes,
    get_simulation_topic_distances,
    get_simulation_pairwise_combinations,
)
from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
    get_semantic_taxonomy,
    get_topic_names,
)


def rec_dd():
    return defaultdict(rec_dd)


@st.cache_data(show_spinner="Fetching histograms with topic counts...")
def get_count_histograms() -> Dict[str, Dict[str, Dict[str, str]]]:
    """Returns count histograms for each taxonomy class and levels 1 to 3.

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]: A dictionary with the
            following structure:
            {
                taxonomy_class: {
                    parent_level: {
                        child_level: histogram_html,
                        ...,
                    },
                    ...,
                },
                ...,
            }
    """

    hists = rec_dd()
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("aria-mapping")
    files_to_get = [
        file.key
        for file in bucket.objects.filter(
            Prefix=f"outputs/validation_figures/journal_histograms/counts/"
        )
        if file.key.endswith(".html")
    ]
    for file in files_to_get:
        obj = bucket.Object(file)
        body = obj.get()["Body"].read().decode("utf-8")
        parsed_fn = file.split("/")[-1].split("_")
        alg_typ = parsed_fn[0]
        top_level = parsed_fn[3]
        next_level = parsed_fn[-1].split(".")[0]
        hists[alg_typ][top_level][next_level] = body
    hists = dict(hists)
    return hists


@st.cache_resource(show_spinner="Fetching histograms with topic frequencies...")
def get_frequency_histograms() -> Dict[str, Dict[str, Dict[str, str]]]:
    """Returns frequency histograms for each taxonomy class and levels 1 to 3.

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]: A dictionary with the
            following structure:
            {
                taxonomy_class: {
                    parent_level: {
                        child_level: histogram_html,
                        ...,
                    },
                    ...,
                },
                ...,
            }
    """
    hists = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict)))
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("aria-mapping")
    files_to_get = [
        file.key
        for file in bucket.objects.filter(
            Prefix=f"outputs/validation_figures/journal_histograms/frequency/"
        )
        if file.key.endswith(".html")
    ]
    for file in files_to_get:
        obj = bucket.Object(file)
        body = obj.get()["Body"].read().decode("utf-8")
        parsed_fn = file.split("/")[-1].split("_")
        alg_typ = parsed_fn[0]
        top_level = parsed_fn[3]
        next_level = parsed_fn[-1].split(".")[0]
        hists[alg_typ][top_level][next_level] = body

    return hists


@st.cache_data(show_spinner="Fetch topic size data")
def get_queried_sizes(
    taxonomies: str, ratios: str, levels: str, simulation: str
) -> pd.DataFrame:
    """Gets topic sizes for a given taxonomy, ratio, and level.

    Args:
        taxonomies (str): A possible list of taxonomy classes to query.
        ratios (str): A possible list of ratios to query.
        levels (str): A list of levels to query.
        simulation (str): The simulation type (noise or sample) to query.

    Returns:
        pd.DataFrame: A dataframe with the size outputs.
    """
    dfs = []
    for alg in taxonomies:
        for rat in ratios:
            df = get_simulation_topic_sizes(alg, rat, simulation)
            df = df.loc[df.level.isin([f"Level_{str(lvl)}" for lvl in levels])]
            if len(taxonomies) > 1:
                df["level"] = df["level"].apply(lambda x: f"{x}_{alg}")
            if len(ratios) > 1:
                df["level"] = df["level"].apply(lambda x: f"{x}_ratio_{rat}")
            dfs.append(df)
    df = pd.concat(dfs)
    df["TopicName"].fillna("No Name", inplace=True)
    return df


@st.cache_data(show_spinner="Fetch topic distance data")
def get_queried_distances(
    taxonomies: str,
    ratios: str,
    levels: str,
    simulation: str,
    entities: Union[str, Sequence[str]],
) -> pd.DataFrame:
    """Gets topic distances for a given taxonomy, ratio, level, and entities.

    Args:
        taxonomies (str): A possible list of taxonomy classes to query.
        ratios (str): A possible list of ratios to query.
        levels (str): A list of levels to query.
        simulation (str): The simulation type (noise or sample) to query.
        entities (Union[str, Sequence[str]]): A list of entities to query.

    Returns:
        pd.DataFrame: A dataframe with the distance outputs.
    """
    dfs = []
    for alg in taxonomies:
        for rat in ratios:
            df = get_simulation_topic_distances(alg, rat, simulation)
            df = (
                df.unstack()
                .reset_index()
                .rename(columns={"level_0": "level", 0: "distance"})
                .assign(level=lambda x: x["level"].str.rstrip("_dist"), taxonomy=alg)
            )
            df = df.loc[df.level.isin([f"Level_{str(lvl)}" for lvl in levels])]
            if len(taxonomies) > 1:
                df["level"] = df["level"].apply(lambda x: f"{x}_{alg}")
            if len(ratios) > 1:
                df["level"] = df["level"].apply(lambda x: f"{x}_ratio_{rat}")
            dfs.append(df)
    df = pd.concat(dfs)

    if entities != "All entities":
        df = df.loc[df.Entity.str.contains("|".join(entities), case=False)]
    return df


@st.cache_data(show_spinner="Getting TS data")
def get_timeseries_data(df: pd.DataFrame, taxonomies: str, ratios: str) -> pd.DataFrame:
    """Gets topic distances TS for a given taxonomy, ratio, level, and entities.

    Args:
        pd.DataFrame: A dataframe with the distance outputs.
        taxonomies (str): A possible list of taxonomy classes to query.
        ratios (str): A possible list of ratios to query.

    Returns:
        pd.DataFrame: A dataframe with the distance outputs for plotting as TS.
    """
    df = df.groupby(["level"], as_index=False).agg({"distance": ["mean", "std"]})
    df["lci"] = df["distance"]["mean"] - df["distance"]["std"]
    df["uci"] = df["distance"]["mean"] + df["distance"]["std"]
    df.columns = ["level", "mean", "std", "lci", "uci"]
    if any([len(taxonomies) > 1, len(ratios) > 1]):
        df[["level", "config"]] = df.level.str.split("(?<=\d)_", expand=True)
    else:
        df["config"] = f"{taxonomies[0]}_ratio_{ratios[0]}"
    return df


@st.cache_data(show_spinner="Fetch pairwise data")
def get_pairwise_data(
    taxonomies: str, ratios: str, simulation: str, topic_selection: str
) -> pd.DataFrame:
    """Gets topic pairwise data for a given taxonomy, ratio, and entities.

    Args:
        taxonomies (str): A possible list of taxonomy classes to query.
        ratios (str): A possible list of ratios to query.
        simulation (str): The simulation type (noise or sample) to query.
        topic_selection (str): The topic selection method to query.

    Returns:
        pd.DataFrame: A dataframe with the pairwise outputs.
    """
    dfs = []
    for alg in taxonomies:
        for rat in ratios:
            pairwise_subtopic_dict = get_simulation_pairwise_combinations(
                alg, rat, simulation, topic_selection
            )

            dfs_ = [
                (
                    pd.DataFrame.from_dict(
                        pairwise_subtopic_dict[topic][subtopic], orient="index"
                    )
                    .reset_index()
                    .rename(columns={"index": "level"})
                    .assign(subtopic=subtopic, topic=topic)
                )
                for topic in pairwise_subtopic_dict.keys()
                for subtopic in pairwise_subtopic_dict[topic].keys()
            ]
            df_ = pd.concat(dfs_)
            df_["ratio"], df_["taxonomy"] = rat, alg
            dfs.append(df_)
    df_topic = pd.concat(dfs)

    dfs = []
    for alg in taxonomies:
        for rat in ratios:
            pairwise_subtopic_dict = get_simulation_pairwise_combinations(
                alg, rat, simulation, topic_selection
            )

            dfs_ = [
                (
                    pd.DataFrame.from_dict(
                        pairwise_subtopic_dict[topic][subtopic][concept],
                        orient="index",
                    )
                    .reset_index()
                    .rename(columns={"index": "level"})
                    .assign(subtopic=subtopic, topic=topic, concept=concept)
                )
                for topic in pairwise_subtopic_dict.keys()
                for subtopic in pairwise_subtopic_dict[topic].keys()
                for concept in pairwise_subtopic_dict[topic][subtopic].keys()
            ]
            df_ = pd.concat(dfs_)
            df_["ratio"], df_["taxonomy"] = rat, alg
            dfs.append(df_)
    df_subtopic = pd.concat(dfs)

    return df_topic, df_subtopic


def get_levels_entity(row: pd.Series):
    """Gets the levels for a given entity.

    Args:
        row (pd.Series): A row from the dataframe.
    """
    level_list = row["Level_5"].split("_")
    level_list.extend(["0"] * (5 - len(level_list)))
    return [int(x) for x in level_list]


@st.cache_data(show_spinner="Fetch taxonomy data")
def get_taxonomy_data() -> Dict[str, pd.DataFrame]:
    """Gets baseline data for a given taxonomy.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary with the taxonomy dataframes.
    """
    dfs = {}
    for taxonomy in ["cooccur", "imbalanced", "centroids"]:
        if taxonomy == "cooccur":
            df = get_cooccurrence_taxonomy()
        elif taxonomy == "imbalanced":
            df = get_semantic_taxonomy(cluster_object="imbalanced")
        elif taxonomy == "centroids":
            df = get_semantic_taxonomy(cluster_object="centroids")

        for level in range(1, 6):
            level_names = get_topic_names(taxonomy, "entity", level, n_top=5, postproc=False)
            df["Level_{}_Entity_Names".format(str(level))] = df[
                "Level_{}".format(str(level))
            ].map(level_names)

        df[["Level_1", "Level_2", "Level_3", "Level_4", "Level_5"]] = df.apply(
            lambda row: get_levels_entity(row), axis=1, result_type="expand"
        )
        dfs[taxonomy] = df
    return dfs
