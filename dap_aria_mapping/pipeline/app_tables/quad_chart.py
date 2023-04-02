"""generates backend of quad chart for change makers tab with disruption and novelty indicators
"""

from dap_aria_mapping.getters.openalex import get_openalex_topics, get_openalex_cd_scores, get_openalex_authorships
from dap_aria_mapping.utils.app_data_utils import expand_topic_col, add_area_domain_chatgpt_names
from dap_aria_mapping.getters.novelty import get_openalex_novelty_scores
import polars as pl
import pandas as pd
from dap_aria_mapping import logger, BUCKET_NAME
import io
import boto3

if __name__ == "__main__":

    logger.info("Loading and aggregating publication cd scores")
    years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    cd_scores = pl.DataFrame(data=None)
    for year in years:
        df = pl.DataFrame(
            pd.DataFrame.from_dict(
                get_openalex_cd_scores(year), orient='index')
            .T
            .unstack()
            .dropna()
            .reset_index(drop=True, level=1)
            .to_frame()
            .reset_index()
        ).with_columns(pl.lit(year).alias("year"))
        cd_scores = pl.concat([cd_scores, df])
    cd_scores.columns = ["id", "cd_score", "year"]
    logger.info("Have c-d scores for {} publications".format(len(cd_scores)))

    logger.info("Loading publication novelty scores")
    novelty_scores = pl.DataFrame(get_openalex_novelty_scores(level=3)).select(pl.col(
        "work_id"), pl.col("novelty"), pl.col("year")).rename({"work_id": "id"})
    logger.info("Have novelty scores for {} publications".format(
        len(novelty_scores)))

    scores_df = cd_scores.join(novelty_scores, on="id", how="outer")

    logger.info("Loading publication authorship info")
    authorships = pl.DataFrame(get_openalex_authorships()
                               ).select([pl.col("id"), pl.col("affiliation_string")]
                                        )

    logger.info("Loading publications with topics")
    pubs_with_topics_df = pl.DataFrame(
        pd.DataFrame.from_dict(
            get_openalex_topics(tax="cooccur", level=3), orient='index')
        .T
        .unstack()
        .dropna()
        .reset_index(drop=True, level=1)
        .to_frame()
        .reset_index()
    )

    pubs_with_topics_df.columns = ["id", "topic"]

    logger.info("Calculating volume, average cd score per organisation per topic, and average novelty score per organisation per topic")

    orgs_df_with_topics_and_scores = pubs_with_topics_df.join(
        authorships, on="id", how="left"
    ).filter(
        ~pl.all(pl.col('affiliation_string').is_null())
    ).join(
        scores_df, on="id", how="left"
    )
    q = (
        orgs_df_with_topics_and_scores.lazy()
        .groupby(["affiliation_string", "topic", "year"])
        .agg([
            pl.count("id").alias("volume"),
            pl.mean("cd_score").alias("average_cd_score"),
            pl.mean("novelty").alias("average_novelty_score")
        ])
    )

    aggregated_data = q.collect()

    logger.info("Adding topic names")

    aggregated_data_with_names = add_area_domain_chatgpt_names(
        expand_topic_col(aggregated_data))

    logger.info("Uploading yearly files to S3")
    years = aggregated_data_with_names[["year"]].unique()
    for year in years:
        yearly_data = aggregated_data_with_names.filter(pl.col("year") == year)
        buffer = io.BytesIO()
        yearly_data.write_parquet(buffer)
        buffer.seek(0)
        s3 = boto3.client("s3")
        s3.upload_fileobj(buffer, BUCKET_NAME,
                          "outputs/app_data/change_makers/disruption/disruption_by_institution_{}.parquet".format(year))
