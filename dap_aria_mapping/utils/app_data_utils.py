"""Utility functions to support pipelines to build tables for the app

"""
import polars as pl
import pandas as pd
from dap_aria_mapping.getters.taxonomies import get_topic_names


def count_documents(doc_df: pl.DataFrame) -> pl.DataFrame:
    """takes a dataframe with documents (including their topic and year) 
    and generates a count of documents per topic per year

    Args:
        doc_df (pl.DataFrame): documents with topics and years

    Returns:
        pl.DataFrame: count of documents per topic per year
    """
    q = (doc_df.lazy().groupby(["topic", "year"]).agg([pl.count()]))
    return q.collect()


def expand_topic_col(topic_df: pl.DataFrame) -> pl.DataFrame:
    """takes a dataframe with a column called "topic" (level 3 of taxonomy), 
        and splits the column to add area ("level 2") and domain ("level 1")

    Args:
        topic_df (pl.DataFrame): dataframe with column for a "topic"

    Returns:
        pl.DataFrame: the same dataframe with "area" and "domain" columns added
    """
    return topic_df.with_columns(
        (pl.col("topic").str.split("_").arr.get(0)).alias("domain"),
        (pl.col("topic").str.split("_").arr.get(0) + "_" + pl.col("topic").str.split("_").arr.get(1)).alias("area"))


def add_area_domain_chatgpt_names(df_with_topic_ids: pl.DataFrame) -> pl.DataFrame:
    """adds area, domain, and topic names to a dataframe tagged with topic, area, and domain ids

    Args:
        df_with_topic_ids (pl.DataFrame): dataframe with id columns named topic, area, domain

    Returns:
        pl.DataFrame: dataframe tagged with topic names
    """

    domain_names = pl.DataFrame(
        pd.DataFrame.from_dict(
            get_topic_names("cooccur", "chatgpt", 1, n_top=35, postproc=True),
            orient="index").rename_axis("domain").reset_index().rename(
                columns={"name": "domain_name"})[["domain", "domain_name"]])
    area_names = pl.DataFrame(
        pd.DataFrame.from_dict(
            get_topic_names("cooccur", "chatgpt", 2, n_top=35, postproc=True),
            orient="index").rename_axis("area").reset_index().rename(
                columns={"name": "area_name"})[["area", "area_name"]])
    topic_names = pl.DataFrame(
        pd.DataFrame.from_dict(
            get_topic_names("cooccur", "chatgpt", 3, n_top=35, postproc=True),
            orient="index").rename_axis("topic").reset_index().rename(
                columns={"name": "topic_name"})[["topic", "topic_name"]])

    df_with_names = df_with_topic_ids.join(
        domain_names, on="domain", how="left")
    df_with_names = df_with_names.join(area_names, on="area", how="left")
    df_with_names = df_with_names.join(topic_names, on="topic", how="left")

    return df_with_names
