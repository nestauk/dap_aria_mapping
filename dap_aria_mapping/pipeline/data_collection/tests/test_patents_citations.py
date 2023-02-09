"""
Tests patent citations collection pipeline.

pytest dap_aria_mapping/pipeline/data_collection/tests/test_patents_citations.py
"""
import pytest
import itertools

from dap_aria_mapping.pipeline.data_collection.patents_citations import (
    base_patents_backward_citation_query,
    base_patents_forward_citation_query,
    chunk_bigquery_query,
    format_citation_data,
    get_citations,
)

import pandas as pd

test_focal_ids = [
    "WO-2007085035-A3",
    "WO-2007138818-A1",
    "WO-2017043448-A1",
    "WO-2017086587-A1",
    "WO-2017125110-A1",
]

backward_citations_dict = {
    "WO-2007085035-A3": ["DE-3823901-A1"],
    "WO-2007138818-A1": ["JP-2006170910-A"],
    "WO-2017043448-A1": [
        "JP-S6044511-A",
        "JP-H11509245-A",
        "JP-2004526048-A",
        "JP-2008502789-A",
        "JP-2011511133-A",
        "JP-2011528401-A",
        "JP-2012503074-A",
    ],
    "WO-2017086587-A1": [
        "KR-20060124069-A",
        "KR-20140023515-A",
        "KR-20140045205-A",
        "EP-2809075-A2",
        "US-2014375823-A1",
    ],
    "WO-2017125110-A1": [
        "DE-102008009361-A1",
        "DE-102008009363-B4",
        "DE-102014008719-B3",
        "DE-102014002731-A1",
    ],
}

forward_citations_dict = {
    "CN-105547945-A": [
        "WO-2007138818-A1",
        "US-2012140223-A1",
        "CN-103674791-A",
        "CN-103868831-A",
        "CN-203705307-U",
        "CN-204514759-U",
        "CN-105241637-A",
    ],
    "CN-106018201-A": [
        "CN-101043581-A",
        "WO-2007138818-A1",
        "CN-101510301-A",
        "CN-102116610-A",
        "CN-102968776-A",
        "CN-103247025-A",
        "CN-103674791-A",
        "CN-105243647-A",
        "CN-105403245-A",
        "CN-105427265-A",
    ],
    "CN-106018201-B": [
        "CN-101043581-A",
        "WO-2007138818-A1",
        "CN-101510301-A",
        "CN-102116610-A",
        "CN-102968776-A",
        "CN-103247025-A",
        "CN-103674791-A",
        "CN-105243647-A",
        "CN-105403245-A",
        "CN-105427265-A",
    ],
}


def test_base_patents_backward_citation_query():
    """
    Test base_patents_backward_citation_query function.
    """
    query = base_patents_backward_citation_query()
    assert type(query) == str


def test_base_patents_forward_citation_query():
    """
    Test base_patents_forward_citation_query function.
    """
    query = base_patents_forward_citation_query()
    assert type(query) == str


def test_chunk_bigquery_query():
    """
    Test chunk_bigquery_query function.
    """
    query = base_patents_backward_citation_query()
    query_chunks = chunk_bigquery_query(
        publication_numbers=test_focal_ids, base_q=query
    )

    assert type(query_chunks) == list
    assert len(query_chunks) == 1
    assert all(t for t in test_focal_ids if t in query_chunks[0])


def test_format_citation_data():
    """
    Test format_citation_data function.
    """
    base_citations_q = base_patents_backward_citation_query()
    backward_citations = get_citations(
        base_citations_q, test_focal_ids, production=False
    )
    formatted_citation_data, formatted_citation_data_dict = format_citation_data(
        backward_citations
    )

    assert type(formatted_citation_data) == pd.DataFrame
    assert len(formatted_citation_data.columns) == 9
    assert type(formatted_citation_data_dict) == dict
    assert len(formatted_citation_data_dict) == len(test_focal_ids)
    assert all(t for t in test_focal_ids if t in formatted_citation_data_dict.keys())


def test_get_citations():
    """
    Test get_citations function.
    """
    base_citations_q = base_patents_backward_citation_query()
    backward_citations = get_citations(
        base_citations_q, test_focal_ids, production=False
    )

    assert type(backward_citations) == pd.DataFrame
    assert len(backward_citations) == len(test_focal_ids)
    assert len(backward_citations.columns) == 2


def test_backward_citations_dict():
    """Tests backward_citations_dict"""

    assert type(backward_citations_dict) == dict
    assert len(backward_citations_dict) == len(test_focal_ids)
    assert all(t for t in test_focal_ids if t in backward_citations_dict.keys())


def test_forward_citations_dict():
    """Tests forward_citations_dict"""
    assert type(forward_citations_dict) == dict
    assert len(
        {
            k: v
            for k, v in forward_citations_dict.items()
            if len(list(set(v) & set(test_focal_ids))) > 0
        }
    ) == len(forward_citations_dict)
