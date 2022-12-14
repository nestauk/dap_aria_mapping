"""
Tests post processing steps for entities pipeline

pytest dap_aria_mapping/pipeline/data_enrichment/tests/test_processed_entities.py
"""
import pytest

from dap_aria_mapping.pipeline.entity_processing.processed_entities import (
    predict_entity_type,
    filter_entities,
    get_all_lookups,
)

entities = {
    "EP-3810804-A1": [
        {"URI": "http://dbpedia.org/resource/Alternative_splicing", "confidence": 80},
        {"URI": "http://dbpedia.org/resource/Genomics", "confidence": 70},
        {"URI": "http://dbpedia.org/resource/Transcriptome", "confidence": 100},
        {"URI": "http://dbpedia.org/resource/Protein", "confidence": 90},
        {"URI": "http://dbpedia.org/resource/RNA", "confidence": 30},
        {"URI": "http://dbpedia.org/resource/Machine_learning", "confidence": 90},
    ],
    "EP-1967857-A3": [],
    "US-2007134705-A1": [
        {"URI": "http://dbpedia.org/resource/Transcription_factor", "confidence": 90},
        {"URI": "http://dbpedia.org/resource/Gene", "confidence": 70},
        {"URI": "http://dbpedia.org/resource/Interval_graph", "confidence": 100},
    ],
    "US-8921074-B2": [
        {"URI": "http://dbpedia.org/resource/Colorectal_cancer", "confidence": 100},
        {"URI": "http://dbpedia.org/resource/Gene", "confidence": 90},
        {"URI": "http://dbpedia.org/resource/IL2RB", "confidence": 90},
        {"URI": "http://dbpedia.org/resource/TSG-6", "confidence": 90},
        {"URI": "http://dbpedia.org/resource/RNA", "confidence": 90},
    ],
    "WO-2020092591-A1": [
        {"URI": "http://dbpedia.org/resource/Germline", "confidence": 70},
        {"URI": "http://dbpedia.org/resource/Genome", "confidence": 90},
        {"URI": "http://dbpedia.org/resource/Allele", "confidence": 90},
    ],
}

entities_list = ["Stanford University", "Allele", "Genome", "5000", "Machine Learning"]


def test_predict_entity_type():

    predicted_entities = predict_entity_type(entities_list)
    assert len(entities_list) == len(entities_list)
    assert (list(set(predicted_entities)) == [0, 1]) == True


def test_filter_entities():
    clean_entities = {
        text_id: filter_entities(ent) for text_id, ent in entities.items()
    }
    assert type(clean_entities) == dict
    assert len(clean_entities) == len(entities)
    assert type(list(clean_entities.keys())[0]) == str
    assert type(list(clean_entities.values())[0]) == list


def test_get_all_lookups():
    links = get_all_lookups()

    assert type(links) == list
    assert all(isinstance(l, str) for l in links)
