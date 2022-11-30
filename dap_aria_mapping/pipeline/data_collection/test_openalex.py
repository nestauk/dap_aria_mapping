import pytest

from dap_aria_mapping.pipeline.data_collection.openalex import (
    api_generator
)

def test_api_generator():
    """Tests the api_generator function"""
    api_root = "https://api.openalex.org/works?filter="
    year = "2018"
    output = api_generator(api_root, year)
    assert len(output) > 1_000
    for url in output:
        assert url.startswith(api_root)
        assert "publication_year:2018" in url
        assert "per-page=200" in url
        assert "cursor=" in url