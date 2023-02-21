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

def get_openalex_topics(tax: str = 'cooccur', level: int = 1) -> Dict:
    """gets openalex document ids tagged with topics from the specified taxonomy
    at a given level"

    Args:
        tax (str, optional): taxonomy labels to use. Defaults to 'cooccur'.
            PIPELINE HAS CURRENTLY ONLY BEEN RUN FOR COOCURRENCE SO THIS IS THE ONLY OPTION
        level (int, optional): level of the taxonomy to use. Defaults to 1.
            options are 1, 2, 3, 4, 5.

    Returns:
        Dict: dictionary with document ids tagged with topic labels of the taxonomy.
            key: openalex id, value: list of topic labels. Labels are formatted as
            [LEVEL 1 LABEL]-[LEVEL 2 LABEL]-[ETC]
    """

    return download_obj(
        BUCKET_NAME,
        "outputs/docs_with_topics/{}/openalex/Level_{}.json".format(tax,str(level)),
        download_as="dict",
    )

#########TEMPORARY AI GENOMICS GETTERS##########################


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


def get_openalex_forward_citations(
    year: int = None,
    focus_papers_only: bool = True,
) -> Dict:
    """_summary_

    Args:
        year (int, optional): The year for which to fetch a subset of forward
            citation data. Defaults to None, which fetches data for all years.
        focus_papers_only (bool, optional): If True, forward citations are only
            fetched for the core works. If False, forward citations are also
            fetched for the works that are cited by the core works. Defaults
            to True.

    Returns:
        Dict: Dictionary mapping work IDs of publications that cite works in
    """
    fp = "focus_papers_only" if focus_papers_only else "expanded"

    if year is not None:
        directory = "inputs/data_collection/processed_openalex/yearly_subsets/"
        fname = f"forward_citations_{fp}_{year}.json"
    else:
        directory = "inputs/data_collection/openalex/"
        fname = f"forward_citations_{fp}.json"

    return download_obj(
        BUCKET_NAME,
        f"{directory}{fname}",
        download_as="dict",
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
