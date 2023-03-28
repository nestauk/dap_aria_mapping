"""generates backend of quad chart for change makers tab with disruption indicator
NOTE: CURRENTLY ONLY INCLUDES PATENTS PUBLISHED IN 2017
"""

from dap_aria_mapping.getters.openalex import get_openalex_topics, get_openalex_cd_scores, get_openalex_authorships
from dap_aria_mapping.utils.app_data_utils import expand_topic_col, add_area_domain_chatgpt_names
import polars as pl
import pandas as pd
from dap_aria_mapping import logger, BUCKET_NAME
import io
import boto3

if __name__ == "__main__":

    logger.info("Loading publication cd scores")
    #NOTE: CURRENTLY ONLY USING 2017 DATA - ONCE PIPELINE IS RUN ON ALL YEARS THIS WILL NEED TO AGGREGATE ALL AVAILABLE YEARS
    cd_scores = pl.DataFrame(
            pd.DataFrame.from_dict(
                get_openalex_cd_scores(year=2017), orient='index')
                .T
                .unstack()
                .dropna()
                .reset_index(drop=True, level=1)
                .to_frame()
                .reset_index()
                )
    cd_scores.columns = ["id", "cd_score"]

    #cd_ids = cd_scores["id"].to_list()

    logger.info("Loading publication authorship info and filtering to only include docs with cd scores")
    authorships = pl.DataFrame(get_openalex_authorships()
        ).select([pl.col("id"), pl.col("affiliation_string")]
        ).filter(pl.col("id").is_in(cd_scores["id"]))

    logger.info("Loading publications with topics and filtering to only include docs with cd scores")
    pubs_with_topics_df = pl.DataFrame(
        pd.DataFrame.from_dict(
            get_openalex_topics(tax = "cooccur", level = 3), orient='index')
            .T
            .unstack()
            .dropna()
            .reset_index(drop=True, level=1)
            .to_frame()
            .reset_index()
            )

    pubs_with_topics_df.columns = ["id", "topic"]

    pubs_with_topics_df = pubs_with_topics_df.filter(pl.col("id").is_in(cd_scores["id"]))

    logger.info("Calculating volume and average cd score per organisation per topic")

    orgs_df_with_topics_and_scores = pubs_with_topics_df.join(
            authorships, on = "id", how = "left"
            ).filter(
                ~pl.all(pl.col('affiliation_string').is_null())
                ).join(
                    cd_scores, on = "id", how = "left"
                )
    q = (
        orgs_df_with_topics_and_scores.lazy()
        .groupby(["affiliation_string", "topic"])
        .agg([
            pl.count("id").alias("volume"),
            pl.mean("cd_score").alias("average_cd_score")
            ])
        )

    aggregated_data = q.collect()

    logger.info("Adding topic names")

    aggregated_data_with_names =  add_area_domain_chatgpt_names(expand_topic_col(aggregated_data))
    
    logger.info("Uploading file to S3")
    buffer = io.BytesIO()
    aggregated_data_with_names.write_parquet(buffer)
    buffer.seek(0)
    s3 = boto3.client("s3")
    s3.upload_fileobj(buffer, BUCKET_NAME, "outputs/app_data/change_makers/disruption_by_institution.parquet")
