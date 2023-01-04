from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME
import pandas as pd


def get_cooccurrence_taxonomy() -> pd.DataFrame:
    """gets taxonomy developed using community detection on term cooccurrence network.
        Algorithm to generate taxonomy can be found in pipeline/taxonomy_development/community_detection.py

    Returns:
        pd.DataFrame: table describing cluster assignments.
        Index: entity, Columns: levels of taxonomy, values are expressed as <INT>_<INT> where there is
        an integer to represent
    """
    return download_obj(
        BUCKET_NAME,
        "outputs/community_detection_clusters.parquet",
        download_as="dataframe",
    )
