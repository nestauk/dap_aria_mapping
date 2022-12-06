"""
Tests that the final patents dataset is filtered appropriately based
    on the filing date being between 2016 - 2021 and that at least
    one inventor is based in GB.

pytest dap_aria_mapping/pipeline/data_collection/tests/test_patents.py
"""
import pytest
from dap_aria_mapping.getters.patents import get_patents

patents_sample = get_patents().sample(50)


def test_patents():
    # make sure every publication number is unique
    assert (
        len(patents_sample) == len(patents_sample["publication_number"].unique())
    ) == True

    # make sure every family id is unique as we should only have one patent document number per family id
    assert (len(patents_sample) == len(patents_sample["family_id"].unique())) == True

    # make sure the filing date is between 2016-01-01 and 2021-12-31
    assert (
        (patents_sample["filing_date"] < "2021-12-31")
        & (patents_sample["filing_date"] > "2016-01-01")
    ).all() == True
    # make sure at least one inventor is from the UK
    assert (
        patents_sample["inventor_harmonized_country_codes"]
        .apply(lambda x: True if "GB" in x else False)
        .all()
        == True
    )
