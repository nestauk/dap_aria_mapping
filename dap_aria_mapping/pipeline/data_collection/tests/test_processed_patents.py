"""
Tests cleaning steps for raw patents data.

pytest dap_aria_mapping/pipeline/data_collection/tests/test_processed_patents.py
"""

import pytest

from dap_aria_mapping.pipeline.data_collection.processed_patents import (
    unnest_column,
    extract_english_text,
)
from dap_aria_mapping import BUCKET_NAME
import pandas as pd

raw_patents_path = "inputs/data_collection/patents/patents_raw.parquet"
patents_sample = pd.read_parquet(f"s3://{BUCKET_NAME}/{raw_patents_path}").sample(50)

INVENTORS = ["JOHN DOE", "MARY JANE", "JACK VINES"]
ASSIGNEES = ["JACK VINES", "UNIV WASHINGTON"]


def test_unnest_column():

    assert "inventor_harmonized" in patents_sample.columns
    assert "assignee_harmonized" in patents_sample.columns

    patents_sample["inventor_harmonized"] = patents_sample["inventor_harmonized"].apply(
        unnest_column
    )
    patents_sample["assignee_harmonized"] = patents_sample["assignee_harmonized"].apply(
        unnest_column
    )

    assert type(patents_sample["inventor_harmonized"].iloc[0]) == tuple
    assert type(patents_sample["assignee_harmonized"].iloc[0]) == tuple

    assert type(patents_sample["inventor_harmonized"].iloc[0][0]) == list
    assert type(patents_sample["assignee_harmonized"].iloc[0][1]) == list

    # based on querying, at least one inventor must have 'GB' associated to them
    assert "GB" in patents_sample["inventor_harmonized"].iloc[1][0]


def test_extract_english_text():
    assert "title_localized" in patents_sample.columns
    assert "abstract_localized" in patents_sample.columns

    patents_sample["title_localized"] = patents_sample["title_localized"].apply(
        extract_english_text
    )
    patents_sample["abstract_localized"] = patents_sample["abstract_localized"].apply(
        extract_english_text
    )

    # based on querying, no abstract should be nan
    assert len(patents_sample[patents_sample["abstract_localized"].isna()]) == 0

    assert type(patents_sample["title_localized"].iloc[0]) == str
    assert type(patents_sample["abstract_localized"].iloc[0]) == str
