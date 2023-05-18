"""
"""
# Script to calculate the total number of patents and publications per topic (level 3), area (level 2), and domain (level 1)
# of the taxonomy per year. The resulting file supports the backend of the Emergence and the Alignment charts in the final app.

from dap_aria_mapping.utils.novelty_utils import create_topic_to_document_dict
from dap_aria_mapping.getters.novelty import (
    get_topic_novelty_openalex,
    get_topic_novelty_patents,
    get_openalex_novelty_scores,
    get_patent_novelty_scores,
    get_openalex_novelty_scores,
)
from dap_aria_mapping.getters.taxonomies import get_topic_names
from dap_aria_mapping.getters.openalex import (
    get_openalex_topics,
    get_openalex_entities,
    get_openalex_works,
)
from dap_aria_mapping.utils.entity_selection import get_sample, filter_entities
from dap_aria_mapping.getters.patents import get_patents, get_patent_topics
from dap_aria_mapping.getters.taxonomies import get_topic_names
from dap_aria_mapping.getters.app_tables.horizon_scanner import get_entities
from dap_aria_mapping.utils.app_data_utils import count_documents, expand_topic_col
import polars as pl
import pandas as pd
from toolz import pipe
from functools import partial
from dap_aria_mapping import BUCKET_NAME, logger
import boto3, pickle, io

LLABELS = {1: "domain", 2: "area", 3: "topic"}


def get_openalex_novelty_df(level: int) -> pl.DataFrame:
    """Transform novelty scores to long format.

    Args:
        level (int): level of taxonomy to use

    Returns:
        pl.DataFrame: long format novelty scores
    """
    return pipe(
        get_openalex_novelty_scores(level=level),
        partial(pl.DataFrame),
        lambda dl: (
            dl.explode("topics")
            .unique(subset=["work_id", "topics"])
            .with_columns(pl.lit(LLABELS[level]).alias("level"))
            .rename({"topics": "topic"})
        ),
        lambda dl: (
            dl.join(
                dl.groupby("topic").agg(pl.count()).rename({"count": "doc_counts"}),
                on="topic",
                how="left",
            )
        ),
    )


if __name__ == "__main__":

    logger.info("Loading documents")
    openalex_df = pl.DataFrame(get_openalex_works())

    logger.info("Loading novelty data")
    openalex_novelty_df = pl.concat(
        [get_openalex_novelty_df(level) for level in [1, 2, 3]]
    )
    # iterate over get_topic_names to create a dictionary. remove rows with "topic" A-Z
    name_dict = {}
    for level in [1, 2, 3]:
        for k, v in get_topic_names("cooccur", "chatgpt", level, n_top=35).items():
            if k not in name_dict:
                name_dict[k] = v["name"]

    names = pl.DataFrame(
        pd.DataFrame.from_dict(
            {**name_dict},
            orient="index",
        )
        .reset_index()
        .rename(columns={"index": "topic", 0: "name"})
    )
    # get top 5 entities
    entities = pl.DataFrame(
        pd.DataFrame.from_dict(
            {
                **{
                    k: v
                    for level in [1, 2, 3]
                    for k, v in get_topic_names(
                        "cooccur", "entity_postproc", level, n_top=5
                    ).items()
                }
            },
            orient="index",
        )
        .reset_index()
        .rename(columns={"index": "topic", 0: "entities"})
    )

    novelty_documents = pipe(
        openalex_novelty_df.unique(subset=["work_id", "topic"]),
        lambda dl: (
            dl[["work_id", "year", "topic", "level", "novelty"]]
            .filter(pl.col("level") == "topic")
            .join(openalex_df[["work_id", "display_name"]], on="work_id", how="left")
        ),
        partial(expand_topic_col),
    )[["work_id", "year", "domain", "area", "topic", "novelty", "display_name"]]

    # Create document-based aggregate novelty metrics
    novelty_trends = pipe(
        # Create in-line results table
        result := (
            openalex_novelty_df.groupby(["topic", "level", "year"]).agg(
                [
                    pl.col("novelty").quantile(0.25).alias("novelty25"),
                    pl.col("novelty").quantile(0.5).alias("novelty50"),
                    pl.col("novelty").quantile(0.75).alias("novelty75"),
                ]
            )
        ),
        # add topic quantiles, create area and domain ids
        lambda dl: (
            dl.filter(pl.col("level") == "topic")
            .rename(
                {
                    "novelty25": "topic_novelty25",
                    "novelty50": "topic_novelty50",
                    "novelty75": "topic_novelty75",
                }
            )
            .drop("level")
        ),
        partial(expand_topic_col),
        # add doc counts
        lambda dl: (
            dl
            # add area and domain means and stds
            .join(
                result.filter(pl.col("level") == "area"),
                left_on=["area", "year"],
                right_on=["topic", "year"],
                how="left",
            )
            .drop("level")
            .rename(
                {
                    "novelty25": "area_novelty25",
                    "novelty50": "area_novelty50",
                    "novelty75": "area_novelty75",
                }
            )
            .join(
                result.filter(pl.col("level") == "domain"),
                left_on=["domain", "year"],
                right_on=["topic", "year"],
                how="left",
            )
            .drop("level")
            .rename(
                {
                    "novelty25": "domain_novelty25",
                    "novelty50": "domain_novelty50",
                    "novelty75": "domain_novelty75",
                }
            )
            # add names & counts
            .join(names, on="topic", how="left")
            .rename({"name": "topic_name"})
            .join(names, left_on="area", right_on="topic", how="left")
            .rename({"name": "area_name"})
            .join(names, left_on="domain", right_on="topic", how="left")
            .rename({"name": "domain_name"})
            .join(entities, on="topic", how="left")
            .rename({"entities": "topic_entities"})
            .join(entities, left_on="area", right_on="topic", how="left")
            .rename({"entities": "area_entities"})
            .join(entities, left_on="domain", right_on="topic", how="left")
            .rename({"entities": "domain_entities"})
            .join(
                openalex_novelty_df[["topic", "doc_counts"]].unique(subset=["topic"]),
                left_on="topic",
                right_on="topic",
                how="left",
            )
            .rename({"doc_counts": "topic_doc_counts"})
            .join(
                openalex_novelty_df[["topic", "doc_counts"]].unique(subset=["topic"]),
                left_on="area",
                right_on="topic",
                how="left",
            )
            .rename({"doc_counts": "area_doc_counts"})
            .join(
                openalex_novelty_df[["topic", "doc_counts"]].unique(subset=["topic"]),
                left_on="domain",
                right_on="topic",
                how="left",
            )
            .rename({"doc_counts": "domain_doc_counts"})
        )[
            [
                "domain",
                "area",
                "topic",
                "year",
                "domain_doc_counts",
                "area_doc_counts",
                "topic_doc_counts",
                "domain_novelty25",
                "domain_novelty50",
                "domain_novelty75",
                "area_novelty25",
                "area_novelty50",
                "area_novelty75",
                "topic_novelty25",
                "topic_novelty50",
                "topic_novelty75",
                "domain_name",
                "area_name",
                "topic_name",
                "domain_entities",
                "area_entities",
                "topic_entities",
            ]
        ],
    )

    logger.info("Uploading file to S3")
    buffer = io.BytesIO()
    novelty_trends.write_parquet(buffer)
    buffer.seek(0)
    s3 = boto3.client("s3")
    s3.upload_fileobj(
        buffer,
        BUCKET_NAME,
        "outputs/app_data/horizon_scanner/agg_novelty_documents.parquet",
    )

    buffer = io.BytesIO()
    novelty_documents.write_parquet(buffer)
    buffer.seek(0)
    s3.upload_fileobj(
        buffer,
        BUCKET_NAME,
        "outputs/app_data/horizon_scanner/novelty_documents.parquet",
    )

    logger.info("Create search list")
    search_df = pipe(
        get_openalex_entities(),
        partial(get_sample, score_threshold=80, num_articles=-1),
        partial(filter_entities, min_freq=10, max_freq=1_000_000, method="absolute"),
    )

    search_df = pl.DataFrame(
        {
            "document_id": [e for e in search_df],
            "entity_list": [search_df[e] for e in search_df],
        }
    )

    # merge work-id unique novelty_documents & display_name with the entity list, and explode
    search_df = search_df.join(
        novelty_documents[["work_id", "display_name"]].unique(subset=["work_id"]),
        left_on="document_id",
        right_on="work_id",
        how="left",
    )

    # create list of all unique entities in nested lists of "entity_list"
    search_list = list(set([e for l in search_df["entity_list"] for e in l]))

    # save pickle list to s3
    buffer = io.BytesIO()
    pickle.dump(search_list, buffer)
    buffer.seek(0)
    s3 = boto3.client("s3")
    s3.upload_fileobj(
        buffer,
        BUCKET_NAME,
        "outputs/app_data/horizon_scanner/entity_list.pkl",
    )

    logger.info("Explode search dataframe")
    search_df = search_df.explode("entity_list")

    buffer = io.BytesIO()
    search_df.write_parquet(buffer)
    buffer.seek(0)
    s3 = boto3.client("s3")
    s3.upload_fileobj(
        buffer,
        BUCKET_NAME,
        "outputs/app_data/horizon_scanner/entity_dataframe.parquet",
    )
