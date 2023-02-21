import pandas as pd
from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME, AI_GENOMICS_BUCKET_NAME
from typing import Mapping, Union, Dict
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

def get_patent_topics(tax: str = 'cooccur', level: int = 1) -> Dict:
    """gets patent ids tagged with topics from the specified taxonomy
    at a given level"

    Args:
        tax (str, optional): taxonomy labels to use. Defaults to 'cooccur'.
            PIPELINE HAS CURRENTLY ONLY BEEN RUN FOR COOCURRENCE SO THIS IS THE ONLY OPTION
        level (int, optional): level of the taxonomy to use. Defaults to 1.
            options are 1, 2, 3, 4, 5.

    Returns:
        Dict: dictionary with document ids tagged with topic labels of the taxonomy.
            key: patent id, value: list of topic labels. Labels are formatted as
            [LEVEL 1 LABEL]-[LEVEL 2 LABEL]-[ETC]
    """

    return download_obj(
        BUCKET_NAME,
        "outputs/docs_with_topics/{}/patents/Level_{}.json".format(tax,str(level)),
        download_as="dict",
    )


#########TEMPORARY AI GENOMICS GETTERS##########################


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
