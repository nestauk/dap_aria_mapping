import pandas as pd
from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME
from typing import Mapping, Union, List, Literal
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


def get_patents_forward_citations(
    year: Literal[2007, 2017] = None,
) -> Mapping[str, List[str]]:
    """From S3 loads dictionary where key is patent id and value is list of forward citations.

    Args:
        year (Literal[2007, 2017], optional): year citations were collected for. Defaults to None.

    Returns:
        Mapping[str, List[str]]: Dictionary where key is patent id and value is list of forward citations.
    """
    directory = "inputs/data_collection/patents/yearly_subsets/"
    fname = f"patents_forward_citations_{year}.json"

    return download_obj(
        BUCKET_NAME,
        f"{directory}{fname}",
        download_as="dict",
    )


def get_patents_backward_citations(
    year: Literal[2007, 2017] = None,
) -> Mapping[str, List[str]]:
    """From S3 loads dictionary where key is patent id and value is list of backward citations.

    Args:
        year (Literal[2007, 2017], optional): year citations were collected for. Defaults to None.

    Returns:
        Mapping[str, List[str]]: Dictionary where key is patent id and value is list of backward citations.
    """

    directory = "inputs/data_collection/patents/yearly_subsets/"
    fname = f"patents_backward_citations_{year}.json"

    return download_obj(
        BUCKET_NAME,
        f"{directory}{fname}",
        download_as="dict",
    )


# not really sure this is especially helpful but it exists
def get_patent_cited_by_citations(
    year: Literal[2007, 2017] = None,
) -> Mapping[str, List[str]]:
    """From S3 loads dictionary where key is patent id and value is list of cited by citations.

    Args:
        year (Literal[2007, 2017], optional): year citations were collected for. Defaults to None.

    Returns:
        Mapping[str, List[str]]: Dictionary where key is patent id and value is list of cited by citations.
    """
    directory = "inputs/data_collection/patents/yearly_subsets/"
    fname = f"patents_cited_by_citations_{year}.json"

    return download_obj(
        BUCKET_NAME,
        f"{directory}{fname}",
        download_as="dict",
    )
