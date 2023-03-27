import pandas as pd
from nesta_ds_utils.loading_saving.S3 import download_obj
from typing import Mapping, Union, Dict, List
from dap_aria_mapping import AI_GENOMICS_BUCKET_NAME, BUCKET_NAME


def get_reliance_on_science() -> pd.DataFrame:
    """Returns dataframe of reliance on science"""
    return download_obj(
        BUCKET_NAME,
        "inputs/data_collection/reliance_on_science/reliance_on_science.parquet",
        download_as="dataframe",
    )
