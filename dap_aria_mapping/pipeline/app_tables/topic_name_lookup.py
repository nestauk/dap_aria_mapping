"""Generates a lookup table of all topic names (level 3) within area names (level 2) within domain names (level 1) using chatgpt names
"""

from dap_aria_mapping.getters.patents import get_patent_topics
from dap_aria_mapping.getters.openalex import get_openalex_topics
from dap_aria_mapping.utils.app_data_utils import expand_topic_col, add_area_domain_chatgpt_names
from nesta_ds_utils.loading_saving.S3 import upload_obj
from dap_aria_mapping import BUCKET_NAME, logger
import polars as pl
import pandas as pd
import io
import boto3


if __name__ == "__main__":
    logger.info("Loading patent data")
    patents_with_topics = get_patent_topics(tax="cooccur", level=3)
    patents_with_topics_df = pl.DataFrame(
        pd.DataFrame.from_dict(patents_with_topics, orient='index'
                               ).T.unstack().dropna().reset_index(drop=True, level=1).to_frame().reset_index())

    patents_with_topics_df.columns = ["publication_number", "topic"]

    logger.info("Loading publication data")
    pubs_with_topics = get_openalex_topics(tax="cooccur", level=3)
    pubs_with_topics_df = pl.DataFrame(
        pd.DataFrame.from_dict(pubs_with_topics, orient='index'
                               ).T.unstack().dropna().reset_index(drop=True, level=1).to_frame().reset_index())
    pubs_with_topics_df.columns = ["publication_number", "topic"]

    logger.info("Creating lookup tables")

    unique_pub_topics = pubs_with_topics_df.unique(subset="topic")
    pub_topics_with_names = add_area_domain_chatgpt_names(
        expand_topic_col(unique_pub_topics)).select((pl.col("domain_name"), pl.col("area_name"), pl.col("topic_name")))

    unique_patent_topics = patents_with_topics_df.unique(subset="topic")
    patent_topics_with_names = add_area_domain_chatgpt_names(
        expand_topic_col(unique_patent_topics)).select((pl.col("domain_name"), pl.col("area_name"), pl.col("topic_name")))

    logger.info("Saving publication lookup table to s3")
    buffer = io.BytesIO()
    pub_topics_with_names.write_parquet(buffer)
    buffer.seek(0)
    s3 = boto3.client("s3")
    s3.upload_fileobj(buffer, BUCKET_NAME,
                      "outputs/app_data/publication_topic_filter_lookup.parquet")

    logger.info("Saving patent lookup table to s3")
    buffer = io.BytesIO()
    patent_topics_with_names.write_parquet(buffer)
    buffer.seek(0)
    s3 = boto3.client("s3")
    s3.upload_fileobj(buffer, BUCKET_NAME,
                      "outputs/app_data/patent_topic_filter_lookup.parquet")
