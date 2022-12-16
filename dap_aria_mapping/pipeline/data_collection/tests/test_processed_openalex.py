import pytest
import json
from pathlib import Path
import pandas as pd

from dap_aria_mapping.pipeline.data_collection.processed_openalex import (
    extract_work_venue,
    make_work_metadata,
    make_work_corpus_metadata,
    make_work_concepts,
    get_authorships,
    make_work_authorships,
    make_citations,
    make_deinverted_abstracts,
    make_inst_metadata,
)

PATH = Path(__file__).parent
EXAMPLE_PUBLICATION = PATH / "test_resources" / "json_string.txt"
INST_META_VARS = [
    "id",
    "display_name",
    "country_code",
    "type",
    "homepage_url"
]
WORK_META_VARS = [
    "id",
    "doi",
    "display_name",
    "publication_year",
    "publication_date",
    "cited_by_count",
    "is_retracted"
]
VENUE_META_VARS = [
    "id",
    "display_name",
    "url"
]

with open(EXAMPLE_PUBLICATION, 'r') as j:
    test_content = json.loads(j.read())

def test_make_work_corpus_metadata():
    assert type(make_work_corpus_metadata(test_content)) == pd.DataFrame
    assert make_work_corpus_metadata(test_content).shape == (1, 10)

def test_make_work_metadata():
    assert type(make_work_metadata(test_content[0])) == dict
    assert len(make_work_metadata(test_content[0]).keys()) == 10

def test_make_citations():
    assert type(make_citations(test_content)) == dict
    assert len(list(make_citations(test_content).values())[0]) ==73

def test_make_deinverted_abstracts():
    assert type(make_deinverted_abstracts(test_content)) == dict
    assert len(list(make_deinverted_abstracts(test_content).values())[0]) == 877

def test_make_work_concepts():
    assert type(make_work_concepts(test_content)) == pd.DataFrame
    assert make_work_concepts(test_content).shape == (20, 4)

def test_make_work_authorships():
    assert type(make_work_authorships(test_content)) == pd.DataFrame
    assert make_work_authorships(test_content).shape == (26, 6)

def test_get_authorships():
    assert type(get_authorships(test_content[0])) == list
    assert len(get_authorships(test_content[0])) == 26

def test_extract_work_venue():
    assert type(extract_work_venue(test_content[0], venue_vars = VENUE_META_VARS)) == dict
    assert len(extract_work_venue(test_content[0], venue_vars = VENUE_META_VARS).keys()) == 3

def test_make_inst_metadata():
    assert type(make_inst_metadata(test_content, meta_vars=INST_META_VARS)) == pd.DataFrame
    assert make_inst_metadata(test_content, meta_vars=INST_META_VARS).shape == (1, 3)


