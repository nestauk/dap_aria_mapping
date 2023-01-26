"""
Tests patent citations collection pipeline.

pytest dap_aria_mapping/pipeline/data_collection/tests/test_patents_citations.py
"""
import pytest

from dap_aria_mapping.pipeline.data_collection.patents_citations import (
    CITATION_MAPPER,
    patents_citation_query,
    patents_forward_citation_query,
)

patent_ids = ["US-2019-0000001-A1", "US-2019-0000002-A1", "US-2019-0000003-A1"]


def test_citation_mapper():

    assert type(CITATION_MAPPER) == dict
    assert all(str(k) for k in CITATION_MAPPER.keys())
    all(str(v) for v in CITATION_MAPPER.values())


def test_patents_citation_query():

    assert type(patents_citation_query(patent_ids)) == str
    assert len(patents_citation_query(patent_ids)) > 0


def test_patents_forward_citation_query():

    assert type(patents_forward_citation_query(patent_ids)) == str
    assert len(patents_forward_citation_query(patent_ids)) > 0
