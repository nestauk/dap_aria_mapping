"""
Tests patent citations collection pipeline.

pytest dap_aria_mapping/pipeline/data_collection/tests/test_patents_citations.py
"""
import pytest

import itertools
from dap_aria_mapping.pipeline.data_collection.patents_citations import (
    CITATION_MAPPER,
    base_patents_citation_query,
    base_patents_backward_citation_query,
    chunk_bigquery_q,
)

focal_ids = ["US-2019-0000001-A1", "US-2019-0000002-A1", "US-2019-0000003-A1"]
forward_citations_list = [
    ["US-2019-0000001-B1", "US-2019-0000002-B1", "US-2019-0000003-B1"],
    ["WO-2019-0000001-A1", "WO-2019-0000002-A1", "WO-2019-0000003-A1"],
    ["ER-2019-0000001-A1", "ER-2019-0000002-A1", "ER-2019-0000003-A1"],
]
flat_forward_citations_list = list(itertools.chain(*forward_citations_list))

backward_citations_list = [
    focal_ids
    + [f"IE-2019-00{i}0001-A1", f"IE-2019-0{i}00002-A1", f"IE-2019-{i}000003-A1"]
    for i in range(len(forward_citations_list))
]

forward_citations = dict(zip(focal_ids, forward_citations_list))
backward_citations = dict(zip(flat_forward_citations_list, backward_citations_list))


def test_citation_mapper():

    assert type(CITATION_MAPPER) == dict
    assert all(str(k) for k in CITATION_MAPPER.keys())
    all(str(v) for v in CITATION_MAPPER.values())


def test_base_patents_citation_query():

    q = base_patents_citation_query()
    assert type(q) == str
    assert q.startswith("SELECT")


def test_base_patents_backward_citation_query():

    q = base_patents_backward_citation_query()
    assert type(q) == str
    assert q.startswith("WITH")


def test_chunk_bigquery_q():

    q = chunk_bigquery_q(focal_ids, base_q=base_patents_citation_query())
    assert type(q) == list
    assert all(type(q) == str for q in q)


def test_patents_backward_citations():

    assert len(
        set(itertools.chain(*list(backward_citations.values()))).intersection(
            set(focal_ids)
        )
    ) == len(focal_ids)
    assert list(backward_citations.keys()) == flat_forward_citations_list


def test_patents_forward_citations():

    assert list(forward_citations.keys()) == focal_ids
