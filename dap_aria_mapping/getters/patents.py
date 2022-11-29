import pandas as pd
from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import AI_GENOMICS_BUCKET_NAME
from typing import Mapping, Union


def get_ai_genomics_patents() -> pd.DataFrame:
    """From S3 loads dataframe of AI in genomics patents
    with columns such as:
        - application_number
        - publication_number
        - full list of cpc codes
        - abstract_text
        - publication_date
        - inventor
        - assignee
    """
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "inputs/patent_data/processed_patent_data/ai_genomics_patents_cpc_codes.csv",
        download_as="dataframe",
    )


def get_ai_sample_patents() -> pd.DataFrame:
    """From S3 loads dataframe of a sample of AI patents (random 10%)
    with columns such as:
        - application_number
        - publication_number
        - full list of cpc codes
        - abstract_text
        - publication_date
        - inventor
        - assignee
    """
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "inputs/patent_data/processed_patent_data/ai_sample_patents_cpc_codes.csv",
        download_as="dataframe",
    )


def get_genomics_sample_patents() -> pd.DataFrame:
    """From S3 loads dataframe of a sample of genomics patents (random 3%)
    with columns such as:
        - application_number
        - publication_number
        - full list of cpc codes
        - abstract_text
        - publication_date
        - inventor
        - assignee
    """
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "inputs/patent_data/processed_patent_data/genomics_sample_patents_cpc_codes.csv",
        download_as="dataframe",
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
