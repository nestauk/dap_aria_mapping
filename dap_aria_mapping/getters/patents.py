import pandas as pd
from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME, AI_GENOMICS_BUCKET_NAME
from typing import Mapping, Union
import pandas as pd


def get_patents() -> pd.DataFrame:
    """From S3 loads dataframe of patents with columns such as:
    - application_number
    - publication_number
    - full list of cpc codes
    - abstract_text
    - publication_date
    - inventor
    - assignee
    """
    return pd.read_parquet(
        f"s3://{BUCKET_NAME}/inputs/data_collection/patents/patents_clean.parquet"
    )


def get_ai_genomics_patents_entities() -> Mapping[str, Mapping[str, Union[str, str]]]:
    """From S3 loads post-processed AI in genomics patents DBpedia entities"""
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "outputs/entity_extraction/ai_genomics_patents_lookup_clean.json",
        download_as="dict",
    )


def get_ai_patents_entities() -> Mapping[str, Mapping[str, Union[str, str]]]:
    """From S3 loads post-processed AI patents DBpedia entities"""
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "outputs/entity_extraction/ai_patents_lookup_clean.json",
        download_as="dict",
    )


def get_genomics_patents_entities() -> Mapping[str, Mapping[str, Union[str, str]]]:
    """From S3 loads post-processed genomics patents DBpedia entities"""
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "outputs/entity_extraction/genomics_patents_lookup_clean.json",
        download_as="dict",
    )
