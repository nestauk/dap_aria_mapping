import json

import pandas as pd
from typing import List, Dict, Any, Mapping, Union
from functools import reduce
from toolz import pipe

from dap_aria_mapping.getters.data_getters import load_s3_data
from dap_aria_mapping import PROJECT_DIR, bucket_name

OALEX_PATH = f"ai_genomics_data/openalex"
PATENTS_PATH = f"ai_genomics_data/patents"

def get_ai_genomics_openalex_works() -> List[Dict[Any, Any]]:
    """Reads OpenAlex works (papers)

    Returns:
        A list of dicts. Every element is a paper with metadata.
        See here for more info: https://docs.openalex.org/about-the-data/work
    """

    return load_s3_data(
        bucket_name,
        f"{OALEX_PATH}/ai_genomics_openalex_works.csv"
    )

def get_ai_genomics_openalex_abstracts() -> list:
    """Reads OpenAlex abstrtacts

    Returns:
        A list of dicts. Every element is a deinverted abstract with metadata.
        See here for more info: https://docs.openalex.org/about-the-data/work
    """

    return load_s3_data(
        bucket_name,
        f"{OALEX_PATH}/ai_genomics_openalex_abstracts.json")

def get_ai_genomics_openalex_ai_genomics_entities() -> Mapping[str, Mapping[str, Union[str, str]]]:
    """From S3 loads ai genomics oa entities

    Returns:
        A list of dicts. Every element is a entity with metadata."""
    return load_s3_data(
        bucket_name,
        f"{OALEX_PATH}/oa_lookup_clean.json",
    )

