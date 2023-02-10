import pandas as pd
from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME
from typing import Mapping, Union, List
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
    return download_obj(
        BUCKET_NAME,
        "inputs/data_collection/patents/patents_clean.parquet",
        download_as="dataframe",
    )


def get_patent_entities() -> Mapping[str, Mapping[str, Union[str, str]]]:
    """From S3 loads patent entities"""
    return download_obj(
        BUCKET_NAME,
        "inputs/data_collection/patents/preprocessed_annotated_patents.json",
        download_as="dict",
    )


def get_patent_forward_citations() -> Mapping[str, List[str]]:
    """From S3 loads patent forward citations"""
    return download_obj(
        BUCKET_NAME,
        "inputs/data_collection/patents/patents_forward_citations.json",
        download_as="dict",
    )


def get_patent_backward_citations() -> Mapping[str, List[str]]:
    """From S3 loads patent backward citations"""
    return download_obj(
        BUCKET_NAME,
        "inputs/data_collection/patents/patents_backward_citations.json",
        download_as="dict",
    )


# not really sure this is especially helpful but it exists
def get_patent_cited_by_citations() -> Mapping[str, List[str]]:
    """From S3 loads patent cited by citations"""
    return download_obj(
        BUCKET_NAME,
        "inputs/data_collection/patents/patents_cited_by_citations.json",
        download_as="dict",
    )
