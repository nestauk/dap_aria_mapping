from dap_aria_mapping.getters.openalex import (
    get_openalex_topics,
    get_openalex_cd_scores,
    get_openalex_authorships,
    get_openalex_works,
)
from dap_aria_mapping.utils.app_data_utils import (
    expand_topic_col,
    add_area_domain_chatgpt_names,
)

from dap_aria_mapping.getters.taxonomies import get_topic_names
from dap_aria_mapping.getters.novelty import get_openalex_novelty_scores
import polars as pl
import pandas as pd
from dap_aria_mapping import logger, BUCKET_NAME
import io
import boto3
from toolz import pipe
from functools import partial

# %%
LLABELS = {1: "domain", 2: "area", 3: "topic"}
# %%
openalex_df = pl.DataFrame(get_openalex_works())
# %%
logger.info("Loading and aggregating publication cd scores")
years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
cd_scores = pl.DataFrame(data=None)
for year in years:
    df = pl.DataFrame(
        pd.DataFrame.from_dict(get_openalex_cd_scores(year), orient="index")
        .T.unstack()
        .dropna()
        .reset_index(drop=True, level=1)
        .to_frame()
        .reset_index()
    ).with_columns(pl.lit(year).alias("year"))
    cd_scores = pl.concat([cd_scores, df])
cd_scores.columns = ["work_id", "cd_score", "year"]
logger.info("Have c-d scores for {} publications".format(len(cd_scores)))
# %%
pubs_with_topics_df = (
    pl.DataFrame(
        pipe(
            get_openalex_topics(tax="cooccur", level=3),
            lambda di: [(k, v) for k, v in di.items() if len(v) > 0],
        ),
        schema=["work_id", "topic"],
    )
    .lazy()
    .explode("topic")
)

pubs_with_topics_df = pubs_with_topics_df.collect()
# %%
pubs_with_topics_df = pipe(
    pubs_with_topics_df.unique(subset=["work_id", "topic"]),
    lambda df: df.join(cd_scores, on="work_id", how="inner"),
    partial(expand_topic_col),
)
# %%
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
# %%
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
# %%
# compute 25, 50 and 75 quantiles of CD scores for year x topic
# and add them to the table
topic_cd_scores = (
    pubs_with_topics_df.groupby(["year", "topic"])
    .agg(
        [
            pl.col("cd_score").quantile(0.25).alias("cd_score25"),
            pl.col("cd_score").quantile(0.5).alias("cd_score50"),
            pl.col("cd_score").quantile(0.75).alias("cd_score75"),
        ]
    )
    .rename(
        {
            "cd_score25": "topic_cd_score25",
            "cd_score50": "topic_cd_score50",
            "cd_score75": "topic_cd_score75",
        }
    )
)

area_cd_scores = (
    pubs_with_topics_df.unique(subset=["work_id", "area"])
    .groupby(["year", "area"])
    .agg(
        [
            pl.col("cd_score").quantile(0.25).alias("cd_score25"),
            pl.col("cd_score").quantile(0.5).alias("cd_score50"),
            pl.col("cd_score").quantile(0.75).alias("cd_score75"),
        ]
    )
    .rename(
        {
            "cd_score25": "area_cd_score25",
            "cd_score50": "area_cd_score50",
            "cd_score75": "area_cd_score75",
        }
    )
)

domain_cd_scores = (
    pubs_with_topics_df.unique(subset=["work_id", "domain"])
    .groupby(["year", "domain"])
    .agg(
        [
            pl.col("cd_score").quantile(0.25).alias("cd_score25"),
            pl.col("cd_score").quantile(0.5).alias("cd_score50"),
            pl.col("cd_score").quantile(0.75).alias("cd_score75"),
        ]
    )
    .rename(
        {
            "cd_score25": "domain_cd_score25",
            "cd_score50": "domain_cd_score50",
            "cd_score75": "domain_cd_score75",
        }
    )
)

topic_doc_counts = (
    pubs_with_topics_df.unique(subset=["work_id", "topic"])
    .groupby("topic")
    .agg(pl.count())
    .rename({"count": "topic_doc_counts"})
)

area_doc_counts = (
    pubs_with_topics_df.unique(subset=["work_id", "area"])
    .groupby("area")
    .agg(pl.count())
    .rename({"count": "area_doc_counts"})
)

domain_doc_counts = (
    pubs_with_topics_df.unique(subset=["work_id", "domain"])
    .groupby("domain")
    .agg(pl.count())
    .rename({"count": "domain_doc_counts"})
)


openalex_disruption_df = (
    pubs_with_topics_df.join(topic_cd_scores, on=["year", "topic"], how="left")
    .join(area_cd_scores, on=["year", "area"], how="left")
    .join(domain_cd_scores, on=["year", "domain"], how="left")
    .unique(subset=["topic", "domain", "area", "year"])
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
    .join(topic_doc_counts, on="topic", how="left")
    .join(area_doc_counts, on="area", how="left")
    .join(domain_doc_counts, on="domain", how="left")
)[
    [
        "domain",
        "area",
        "topic",
        "year",
        "domain_cd_score25",
        "domain_cd_score50",
        "domain_cd_score75",
        "area_cd_score25",
        "area_cd_score50",
        "area_cd_score75",
        "topic_cd_score25",
        "topic_cd_score50",
        "topic_cd_score75",
        "domain_name",
        "area_name",
        "topic_name",
        "domain_entities",
        "area_entities",
        "topic_entities",
        "domain_doc_counts",
        "area_doc_counts",
        "topic_doc_counts",
    ]
]

# %%
logger.info("Uploading all files to S3")

buffer = io.BytesIO()
openalex_disruption_df.write_parquet(buffer)
buffer.seek(0)
s3 = boto3.client("s3")
s3.upload_fileobj(
    buffer,
    BUCKET_NAME,
    "outputs/app_data/horizon_scanner/agg_disruption_documents.parquet",
)

# %%
buffer = io.BytesIO()
pubs_with_topics_df.write_parquet(buffer)
buffer.seek(0)
s3.upload_fileobj(
    buffer,
    BUCKET_NAME,
    "outputs/app_data/horizon_scanner/disruption_documents.parquet",
)
# %%
