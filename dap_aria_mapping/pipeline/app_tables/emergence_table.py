"""Script to calculate the total number of patents and publications per topic (level 3), area (level 2), and domain (level 1) 
    of the taxonomy per year. The resulting file supports the backend of the Emergence and the Alignment charts in the final app.
"""
from dap_aria_mapping.getters.openalex import get_openalex_topics, get_openalex_works
from dap_aria_mapping.getters.patents import get_patent_topics, get_patents
from dap_aria_mapping.utils.app_data_utils import count_documents, expand_topic_col
import polars as pl
import pandas as pd
import boto3
import io
from dap_aria_mapping.getters.taxonomies import get_topic_names
from dap_aria_mapping import BUCKET_NAME, logger

if __name__ == "__main__":

    logger.info("Loading patent data")
    patents_with_topics = get_patent_topics(tax = "cooccur", level = 3)

    logger.info("Transforming patent dictionary to polars df")
    patents_with_topics_df = pl.DataFrame(
        pd.DataFrame.from_dict(patents_with_topics, orient='index'
        ).T.unstack().dropna().reset_index(drop=True, level=1).to_frame().reset_index())
    patents_with_topics_df.columns = ["id", "topic"]

    logger.info("Loading patent date data")
    patent_dates = pl.DataFrame(get_patents()).select(
        ["publication_number", "publication_date"]
        ).with_columns(
        pl.col("publication_date").dt.year().cast(pl.Int64, strict=False).alias("year")
        ).rename(
            {"publication_number": "id"}
            )[["id", "year"]]

    logger.info("Generating patent volume data")
    patents_df = patents_with_topics_df.join(patent_dates, how = 'left', on = "id")
    patents_count = count_documents(patents_df)
    patents_count.columns = ['topic', 'year', 'patent_count']
    final_patents_df = expand_topic_col(patents_count)

    
    logger.info("Loading publication data")
    pubs_with_topics = get_openalex_topics(tax = "cooccur", level = 3)

    logger.info("Transforming publication dictionary to polars df")
    pubs_with_topics_df = pl.DataFrame(pd.DataFrame.from_dict(pubs_with_topics, orient='index').T.unstack().dropna().reset_index(drop=True, level=1).to_frame().reset_index())
    pubs_with_topics_df.columns = ["id", "topic"]

    logger.info("Loading publication date data")
    pub_dates = pl.DataFrame(get_openalex_works()).select(["work_id", "publication_year"]).rename({"publication_year": "year", "work_id": "id"})

    logger.info("Generating publication volume data")

    pubs_df = pubs_with_topics_df.join(pub_dates, how = 'left', on = "id")
    pubs_count = count_documents(pubs_df)
    pubs_count.columns = ['topic', 'year', 'publication_count']
    final_pubs_df = expand_topic_col(pubs_count)
    
    logger.info("Creating merged table")
    final_df = final_pubs_df.join(final_patents_df, how='outer', on = ["domain", "area", "topic", "year"]).with_columns([
        pl.col("publication_count").fill_null(
            pl.lit(0),
        ),
        pl.col("patent_count").fill_null(
            pl.lit(0))
    ])

    logger.info("Adding total document count and topic names columns")
    #add total document count as count of patents and publications combined
    final_df = final_df.with_columns(
        (pl.col('patent_count') + pl.col('publication_count')).alias('total_docs'),
        pl.col('year').round(0)
    )

    #add chatgpt names for domain, area, topics
    domain_names  = pl.DataFrame(
        pd.DataFrame.from_dict(
            get_topic_names("cooccur", "chatgpt", 1, n_top = 35), 
            orient= "index").rename_axis("domain").reset_index().rename(
                columns = {"name": "domain_name"})[["domain", "domain_name"]])
    area_names  = pl.DataFrame(
        pd.DataFrame.from_dict(
            get_topic_names("cooccur", "chatgpt", 2, n_top = 35),
            orient= "index").rename_axis("area").reset_index().rename(
                columns = {"name": "area_name"})[["area", "area_name"]])
    topic_names  = pl.DataFrame(
        pd.DataFrame.from_dict(
            get_topic_names("cooccur", "chatgpt", 3, n_top = 35),
            orient= "index").rename_axis("topic").reset_index().rename(
                columns = {"name": "topic_name"})[["topic", "topic_name"]])

    final_df = final_df.join(domain_names, on="domain", how="left")
    final_df = final_df.join(area_names, on="area", how="left")
    final_df = final_df.join(topic_names, on="topic", how="left")

    logger.info("Uploading file to S3")
    buffer = io.BytesIO()
    final_df.write_parquet(buffer)
    buffer = buffer.seek(0)
    s3 = boto3.client("s3")
    s3.upload_fileobj(buffer, BUCKET_NAME, "outputs/app_data/horizon_scanner/volume.parquet")



    