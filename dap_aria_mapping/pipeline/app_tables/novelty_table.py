"""
"""
# Script to calculate the total number of patents and publications per topic (level 3), area (level 2), and domain (level 1)
# of the taxonomy per year. The resulting file supports the backend of the Emergence and the Alignment charts in the final app.
# %%
from dap_aria_mapping.utils.novelty_utils import create_topic_to_document_dict
from dap_aria_mapping.getters.novelty import (
    get_topic_novelty_openalex,
    get_topic_novelty_patents,
    get_openalex_novelty_scores,
    get_patent_novelty_scores,
)
from dap_aria_mapping.getters.taxonomies import get_topic_names
from dap_aria_mapping.getters.openalex import get_openalex_topics, get_openalex_works
from dap_aria_mapping.getters.patents import get_patents, get_patent_topics
from dap_aria_mapping.getters.taxonomies import get_topic_names
from dap_aria_mapping.utils.app_data_utils import count_documents, expand_topic_col
import polars as pl
import pandas as pd
from toolz import pipe
from functools import partial
from dap_aria_mapping import BUCKET_NAME, logger
import boto3
import io

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
    )


if __name__ == "__main__":
    logger.info("Loading novelty data")
    # iterate over levels and concatenate the polars dataframes
    openalex_novelty_df = pl.concat(
        [get_openalex_novelty_df(level) for level in [1, 2, 3]]
    )

    # iterate over get_topic_names to create a dictionary
    names = pl.DataFrame(
        pd.DataFrame.from_dict(
            {
                **{
                    k: v["name"]
                    for level in [1, 2, 3]
                    for k, v in get_topic_names(
                        "cooccur", "chatgpt", level, n_top=35
                    ).items()
                }
            },
            orient="index",
        )
        .reset_index()
        .rename(columns={"index": "topic", 0: "name"})
    )

    # rename column
    novelty_trends = pipe(
        # Create in-line results table
        result := (
            openalex_novelty_df.groupby(["topic", "level", "year"]).agg(
                [
                    pl.col("novelty").mean().alias("mean"),
                    pl.col("novelty").std().alias("std"),
                ]
            )
        ),
        # add topic means and stds, create area and domain ids
        lambda dl: (
            dl.filter(pl.col("level") == "topic")
            .rename({"mean": "topic_mean", "std": "topic_std"})
            .drop("level")
        ),
        partial(expand_topic_col),
        # add area and domain means and stds
        lambda dl: (
            dl.join(
                result.filter(pl.col("level") == "area"),
                left_on=["area", "year"],
                right_on=["topic", "year"],
                how="left",
            )
            .drop("level")
            .rename({"mean": "area_mean", "std": "area_std"})
            .join(
                result.filter(pl.col("level") == "domain"),
                left_on=["domain", "year"],
                right_on=["topic", "year"],
                how="left",
            )
            .drop("level")
            .rename({"mean": "domain_mean", "std": "domain_std"})
            # add names
            .join(names, on="topic", how="left")
            .rename({"name": "topic_name"})
            .join(names, left_on="area", right_on="topic", how="left")
            .rename({"name": "area_name"})
            .join(names, left_on="domain", right_on="topic", how="left")
            .rename({"name": "domain_name"})
        )[
            [
                "domain",
                "area",
                "topic",
                "year",
                "domain_mean",
                "domain_std",
                "area_mean",
                "area_std",
                "topic_mean",
                "topic_std",
                "domain_name",
                "area_name",
                "topic_name",
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


# %%

# Create a single dataframe in the end where you have the topic IDs for levels 1 to 3, with 1 (and 2) duplicated if they split in different branches in 3.

# %%

# patents_with_topics_df = pl.DataFrame(
#     pd.DataFrame.from_dict(get_patent_topics(tax = "cooccur", level = 3), orient='index')
#     .T
#     .unstack()
#     .dropna()
#     .reset_index(drop=True, level=1)
#     .to_frame()
#     .reset_index()
#     .rename({"index": "id", 0: "topic"}, axis = 1)
# )
# # topic_novelty_openalex_df = get_topic_novelty_openalex()
# # %%
# patent_dates = (
#     pl.DataFrame(
#         get_patents()
#     )
#     .select(["publication_number", "publication_date"])
#     .with_columns(
#         pl.col("publication_date").dt.year().cast(pl.Int64, strict=False).alias("year")
#     )
#     .rename({"publication_number": "id"})
#     [["id", "year"]]
# )


# %%
# if __name__ == "__main__":


#     logger.info("Generating patent volume data")
#     patents_df = patents_with_topics_df.join(patent_dates, how = 'left', on = "id")
#     patents_count = count_documents(patents_df)
#     patents_count.columns = ['topic', 'year', 'patent_count']
#     final_patents_df = expand_topic_col(patents_count)


#     logger.info("Loading publication data")
#     pubs_with_topics = get_openalex_topics(tax = "cooccur", level = 3)

#     logger.info("Transforming publication dictionary to polars df")
#     pubs_with_topics_df = pl.DataFrame(pd.DataFrame.from_dict(pubs_with_topics, orient='index').T.unstack().dropna().reset_index(drop=True, level=1).to_frame().reset_index())
#     pubs_with_topics_df.columns = ["id", "topic"]

#     logger.info("Loading publication date data")
#     pub_dates = pl.DataFrame(get_openalex_works()).select(["work_id", "publication_year"]).rename({"publication_year": "year", "work_id": "id"})

#     logger.info("Generating publication volume data")

#     pubs_df = pubs_with_topics_df.join(pub_dates, how = 'left', on = "id")
#     pubs_count = count_documents(pubs_df)
#     pubs_count.columns = ['topic', 'year', 'publication_count']
#     final_pubs_df = expand_topic_col(pubs_count)

#     logger.info("Creating merged table")
#     final_df = final_pubs_df.join(final_patents_df, how='outer', on = ["domain", "area", "topic", "year"]).with_columns([
#         pl.col("publication_count").fill_null(
#             pl.lit(0),
#         ),
#         pl.col("patent_count").fill_null(
#             pl.lit(0))
#     ])

#     logger.info("Adding total document count and topic names columns")
#     #add total document count as count of patents and publications combined
#     final_df = final_df.with_columns(
#         (pl.col('patent_count') + pl.col('publication_count')).alias('total_docs')
#     )

#     #add chatgpt names for domain, area, topics
#     domain_names  = pl.DataFrame(
#         pd.DataFrame.from_dict(
#             get_topic_names("cooccur", "chatgpt", 1, n_top = 35),
#             orient= "index").rename_axis("domain").reset_index().rename(
#                 columns = {"name": "domain_name"})[["domain", "domain_name"]])
#     area_names  = pl.DataFrame(
#         pd.DataFrame.from_dict(
#             get_topic_names("cooccur", "chatgpt", 2, n_top = 35),
#             orient= "index").rename_axis("area").reset_index().rename(
#                 columns = {"name": "area_name"})[["area", "area_name"]])
#     topic_names  = pl.DataFrame(
#         pd.DataFrame.from_dict(
#             get_topic_names("cooccur", "chatgpt", 3, n_top = 35),
#             orient= "index").rename_axis("topic").reset_index().rename(
#                 columns = {"name": "topic_name"})[["topic", "topic_name"]])

#     final_df = final_df.join(domain_names, on="domain", how="left")
#     final_df = final_df.join(area_names, on="area", how="left")
#     final_df = final_df.join(topic_names, on="topic", how="left")

#     logger.info("Uploading file to S3")
#     buffer = io.BytesIO()
#     final_df.write_parquet(buffer)
#     buffer.seek(0)
#     s3 = boto3.client("s3")
#     s3.upload_fileobj(buffer, BUCKET_NAME, "outputs/app_data/horizon_scanner/volume.parquet")
