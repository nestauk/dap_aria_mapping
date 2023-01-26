"""
OpenAlex getters.

note:
    The data also can include columns we added:
        - scope related columns (genomics_in_scope, genomics_tangential) - this
            is related to a validation exercise conducted as part of AI genomics.
            It should be ignored for these purposes.
        -  ai, genomics, ai_genomics columns- this was an exercise to identify
            research papers at the intersection of ai (ai == True) and genomics
            (genomics == True). These columns should also be ignored.
"""
import pandas as pd
from nesta_ds_utils.loading_saving.S3 import download_obj
from typing import Mapping, Union, Dict
from dap_aria_mapping import AI_GENOMICS_BUCKET_NAME, BUCKET_NAME


def get_openalex_abstracts() -> Dict:
    """Returns a dictionary of AI in genomics OpenAlex abstracts where the key
    is the work_id and the value is the abstract associated to the work id."""
    return download_obj(
        BUCKET_NAME,
        "inputs/data_collection/processed_openalex/abstracts.json",
        download_as="dict",
    )


def get_openalex_works() -> pd.DataFrame:
    """Returns dataframe of OpenAlex works. Abstracts are NOT included.
    Works data includes:
        - work_id;
        - doi;
        - display_name;
        - publication_year;
        - publication_date;
        - cited_by_count
    """
    return download_obj(
        BUCKET_NAME,
        "inputs/data_collection/processed_openalex/works.parquet",
        download_as="dataframe",
    )


def get_openalex_concepts() -> pd.DataFrame:
    """Returns dataframe of concepts related to works in OpenAlex."""
    return download_obj(
        BUCKET_NAME,
        "inputs/data_collection/processed_openalex/concepts.parquet",
        download_as="dataframe",
    )


def get_openalex_authorships() -> pd.DataFrame:
    """From S3 loads openalex authorships information"""
    return download_obj(
        BUCKET_NAME,
        "inputs/data_collection/processed_openalex/authorships.parquet",
        download_as="dataframe",
    )


def get_openalex_citations() -> Dict:
    """From S3 loads openalex citations information"""
    return download_obj(
        BUCKET_NAME,
        "inputs/data_collection/processed_openalex/citations.json",
        download_as="dict",
    )


def get_openalex_entities() -> Mapping[str, Mapping[str, Union[str, str]]]:
    """From S3 loads openalex entities"""
    return download_obj(
        BUCKET_NAME,
        "inputs/data_collection/processed_openalex/preprocessed_annotated_abstracts.json",
        download_as="dict",
    )
