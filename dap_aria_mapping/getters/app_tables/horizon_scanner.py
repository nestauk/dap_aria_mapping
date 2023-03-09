import polars as pl
from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME

def volume_per_year() -> pl.DataFrame:
    """gets a polars dataframe with the count of patents and publications 
    per topic/area/domain per year

    Returns:
        pl.DataFrame: count of patents and publications 
    per topic/area/domain per year
    """
    return pl.DataFrame(download_obj(
        BUCKET_NAME,
        "outputs/app_data/horizon_scanner/volume.parquet",
        download_as="dataframe",
    ))