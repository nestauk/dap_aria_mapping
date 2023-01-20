import json
from pathlib import Path
import pytest

from dap_aria_mapping.pipeline.data_collection.openalex_forward_citations import (
    parse_results,
)


PATH = Path(__file__).parent


TEST_API_OUTPUT = PATH / "test_resources" / "json_citations_string.txt"
TEST_PARSED_OUTPUT = PATH / "test_resources" / "json_citations_output.json"

with open(TEST_API_OUTPUT, "r") as f:
    test_response = json.loads(f.read())

with open(TEST_PARSED_OUTPUT, "r") as f:
    test_parsed = json.load(f)


def test_parse_results():
    assert parse_results(test_response) == test_parsed
