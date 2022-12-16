import pytest
import numpy as np

from dap_aria_mapping.pipeline.entity_processing.reformat_entities import (
    reformat_dictionary
)


EXAMPLE_INPUT = {
    "a":[],
    "b": [
        {
            "URI": "http://dpbedia.org/resource/Heat",
            "confidence": 70
        },
        {
            "URI": "http://dpbedia.org/resource/Energy_consumption",
            "confidence": 80
        }
    ],
    "c": [
        {
            "URI": "http://dpbedia.org/resource/Heat",
            "confidence": 40
        },
        {
            "URI": "http://dpbedia.org/resource/Energy_consumption",
            "confidence": 40
        }
    ]
}

def test_reformat_dictionary():
    assert reformat_dictionary(EXAMPLE_INPUT) == {
        "a": [],
        "b": [
            {
                "entity": "Heat",
                "confidence": 70
            },
            {
                "entity": "Energy consumption",
                "confidence": 80
            }
        ],
        "c": []
    }
