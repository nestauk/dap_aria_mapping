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
from dap_aria_mapping import AI_GENOMICS_BUCKET_NAME


def get_openalex_abstracts() -> list:
    """Returns a dictionary of AI in genomics OpenAlex abstracts where the key
    is the work_id and the value is the abstract associated to the work id."""
    return download_obj(
        "aria-mapping",
        "inputs/data_collection/processed_openalex/abstracts.json",
        download_as="list",
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
        "aria-mapping",
        "inputs/data_collection/processed_openalex/works.json",
        download_as="dataframe",
    )


def get_openalex_concepts() -> pd.DataFrame:
    """Returns dataframe of concepts related to works in OpenAlex."""
    return download_obj(
        "aria-mapping",
        "inputs/data_collection/processed_openalex/concepts.parquet",
        download_as="dataframe",
    )


def get_openalex_authorships() -> pd.DataFrame:
    """From S3 loads openalex authorships information"""
    return download_obj(
        "aria-mapping",
        "outputs/openalex/openalex_authorships.csv",
        download_as="dataframe",
    )


def get_openalex_citations() -> pd.DataFrame:
    """From S3 loads openalex citations information"""
    return download_obj(
        "aria-mapping",
        "outputs/openalex/openalex_citations.json",
        download_as="list",
    )


def get_openalex_ai_genomics_works() -> pd.DataFrame:
    """Returns dataframe of AI in genomics OpenAlex works. Abstracts are NOT included.
    Works data includes:
        - work_id;
        - doi;
        - display_name;
        - publication_year;
        - publication_date;
        - cited_by_count
    """
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "outputs/openalex/ai_genomics_openalex_works.csv",
        download_as="dataframe",
    )


def get_openalex_ai_genomics_abstracts() -> Dict[str, str]:
    """Returns a dictionary of AI in genomics OpenAlex abstracts where the key
    is the work_id and the value is the abstract associated to the work id."""
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "outputs/openalex/ai_genomics_openalex_abstracts.json",
        download_as="dict",
    )


def get_openalex_ai_works() -> pd.DataFrame:
    """Returns dataframe of AI OpenAlex works. Abstracts are NOT included.
    Works data includes:
        - work_id;
        - doi;
        - display_name;
        - publication_year;
        - publication_date;
        - cited_by_count
    """
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "outputs/openalex/ai_openalex_works.csv",
        download_as="dataframe",
    )


def get_openalex_ai_abstracts() -> Dict[str, str]:
    """Returns a dictionary of AI OpenAlex abstracts where the key
    is the work_id and the value is the abstract associated to the work id."""
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "outputs/openalex/ai_openalex_abstracts.json",
        download_as="dict",
    )


def get_openalex_genomics_works() -> pd.DataFrame:
    """Returns dataframe of genomics OpenAlex works. Abstracts are NOT included.
    Works data includes:
        - work_id;
        - doi;
        - display_name;
        - publication_year;
        - publication_date;
        - cited_by_count
    """
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "outputs/openalex/genomics_openalex_works.csv",
        download_as="dataframe",
    )


def get_openalex_genomics_abstracts() -> Dict[str, str]:
    """Returns a dictionary of genomics OpenAlex abstracts where the key
    is the work_id and the value is the abstract associated to the work id."""
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "outputs/openalex/genomics_openalex_abstracts.json",
        download_as="dict",
    )


def get_openalex_entities() -> Mapping[str, Mapping[str, Union[str, str]]]:
    """From S3 loads post-processed openalex DBpedia entities"""
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "outputs/entity_extraction/oa_lookup_clean.json",
        download_as="dict",
    )


def get_openalex_institutes() -> pd.DataFrame:
    """From S3 loads institutes information for work ids, including:
    - work_id;
    - auth_display_name;
    - auth_orcid;
    - affiliation_string;
    - inst_id;
    - year
    """
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "outputs/openalex/openalex_institutes.csv",
        download_as="dataframe",
    )


# We didn't use this table - it looks like it would need cleaning up if you were keen
# to use meSH information
def get_openalex_mesh() -> pd.DataFrame:
    """From S3 loads openalex mesh information, including:
    - word_id;
    - meSH code;
    """
    return download_obj(
        AI_GENOMICS_BUCKET_NAME,
        "outputs/openalex/openalex_mesh.csv",
        download_as="dataframe",
    )
