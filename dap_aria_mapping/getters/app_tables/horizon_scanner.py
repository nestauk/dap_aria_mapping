import polars as pl
from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME


def volume_per_year() -> pl.DataFrame:
    """gets a polars dataframe with the count of patents and publications
    per topic/area/domain per year

    Returns:
        pl.DataFrame: count of patents and publications per topic/area/domain per year.
            Contains columns:
                domain: id associated with level 1 topic
                domain_name: chat_gpt name associated with level 1 topic
                area: id associated with level 2 topic
                area_name: chat_gpt name associated with level 2 topic
                topic: id associated with level 3 topic
                topic_name: chat_gpt name associated with level 3 topic
                year: year of publication of patent/publication
                publication_count: total publications per topic/area/domain per year
                patent_count: total patents per topic/area/domain per year
                total_docs: sum of publication count and patent count

    """
    return pl.DataFrame(
        download_obj(
            BUCKET_NAME,
            "outputs/app_data/horizon_scanner/volume.parquet",
            download_as="dataframe",
        )
    )


def novelty_per_year() -> pl.DataFrame:
    """Gets a polars dataframe with the novelty scores per topic/area/domain per year

    Returns:
        pl.DataFrame: A polars dataframe.
    """
    return pl.DataFrame(
        download_obj(
            BUCKET_NAME,
            "outputs/app_data/horizon_scanner/agg_novelty_documents.parquet",
            download_as="dataframe",
        )
    )
