import pytest

from dap_aria_mapping.pipeline.entity_processing.reformat_entities import (
    reformat_dictionary
)


EXAMPLE_DICT_A = {
    "id": "US-9726461-B2",
    "abstract": "A terrain disruption device"
}

EXAMPLE_DICT_B = {
    'id': 'US-9792517-B2',
    'abstract': 'A computer-implemented method ',
    'dbpedia_entities': [
        {'confidence': 70, 'URI': 'vector graphics'},
        {'confidence': 60, 'URI': 'stroke'},
        {'confidence': 70, 'URI': 'pointing device'},
        {'confidence': 80, 'URI': 'real-time computer graphics'}],
    'dbpedia_entities_metadata': {
        'dupes_10_count': 8,
        'dupes_60_count': 2,
        'entities_count': 30,
        'confidence_counts': {
            '20': 1,
            '30': 24,
            '40': 1,
            '60': 1,
            '70': 2,
            '80': 1},
        'confidence_max': 80,
        'dupes_10_ratio': 0.26666666666666666,
        'dupes_60_ratio': 0.06666666666666667,
        'confidence_avg': 35.333333333333336,
        'confidence_min': 20}
    }


EXAMPLE_DICT_C = {
    'id': 'US-9792517-B2',
    'abstract': 'A computer-implemented method ',
    'dbpedia_entities': [
        {'confidence': 20, 'URI': 'vector graphics'},
        {'confidence': 20, 'URI': 'stroke'},
        {'confidence': 20, 'URI': 'pointing device'},
        {'confidence': 20, 'URI': 'real-time computer graphics'}],
    'dbpedia_entities_metadata': {
        'dupes_10_count': 8,
        'dupes_60_count': 2,
        'entities_count': 30,
        'confidence_counts': {
            '20': 1,
            '30': 24,
            '40': 1,
            '60': 1,
            '70': 2,
            '80': 1},
        'confidence_max': 80,
        'dupes_10_ratio': 0.26666666666666666,
        'dupes_60_ratio': 0.06666666666666667,
        'confidence_avg': 35.333333333333336,
        'confidence_min': 20}
    }

def test_reformat_dictionary():
    assert reformat_dictionary(EXAMPLE_DICT_A) == {
        "id": "US-9726461-B2",
        "dbpedia_entities": []
    }
    assert reformat_dictionary(EXAMPLE_DICT_B) == {
        "id": "US-9792517-B2",
        "dbpedia_entities": [
            {
                "confidence": 70,
                "URI": "vector graphics"
            },
            {
                "confidence": 60,
                "URI": "stroke"
            },
            {
                "confidence": 70,
                "URI": "pointing device"
            },
            {
                "confidence": 80,
                "URI": "real-time computer graphics"
            }
        ]
    }
    assert reformat_dictionary(EXAMPLE_DICT_C) == {
        "id": "US-9792517-B2",
        "dbpedia_entities": []
    }
