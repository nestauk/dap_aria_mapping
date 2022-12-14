"""
This one-off script generates a validation sample of entities extracted
    from patents to determine the incidence of entity
    false positives for entity pre- and post- processing.

python dap_aria_mapping/pipeline/entity_processing/evaluation/generate_validation_sample.py
"""
from dap_aria_mapping import BUCKET_NAME
from dap_aria_mapping.pipeline.entity_processing.process_entities import (
    predict_entity_type,
)
from dap_aria_mapping.getters.patents import get_patents

from nesta_ds_utils.loading_saving.S3 import download_obj, upload_obj
import random
from typing import List
import pandas as pd

ANNOTATED_PATENTS_PATH = (
    "inputs/data_collection/patents/preprocessed_annotated_patents.json"
)
VALIDATION_SAMPLE_PATH = "inputs/data_collection/entity_validation_sample.csv"


def _make_entity_sample(dict_list: List[dict]) -> List[dict]:
    """
    Makes a sample of 100 non-empty unique documents, their
        associated extracted entities, confidence scores and
        post-processed spaCy labels
    """
    nonempty_dict_list = []
    for ent in dict_list:
        id_ = ent["id"]
        entities = [
            e["URI"].split("/")[-1].replace("_", " ") for e in ent["dbpedia_entities"]
        ]
        confidence_scores = [e["confidence"] for e in ent["dbpedia_entities"]]
        spacy_ents = predict_entity_type(entities)
        for i, e in enumerate(entities):
            nonempty_dict_list.append([id_, e, confidence_scores[i], spacy_ents[i]])

    return nonempty_dict_list


def get_entity_sample(dict_list: List[dict], data_source: str) -> pd.DataFrame:
    """
    Gets a sample of 100 non-empty unique documents, converts to
        a dataframe and adds a column defining the data source

    Returns a dataframe of document ids, list of entities, a
        list of confidence scores and the data source name
    """

    ent_sample = _make_entity_sample(dict_list)

    return (
        pd.DataFrame(ent_sample)
        .rename(columns={0: "id", 1: "entity", 2: "confidence_score", 3: "spacy_label"})
        .assign(data_source=data_source)
    )


if __name__ == "main":

    random.seed(42)

    patent_ents = download_obj(BUCKET_NAME, ANNOTATED_PATENTS_PATH, download_as="dict")
    nonempty_patents = [ent for ent in patent_ents if ent["dbpedia_entities"] != []]
    random.shuffle(nonempty_patents)
    nonempty_patents_sample = nonempty_patents[:100]

    patents = get_patents()
    patents_sample_df = get_entity_sample(
        nonempty_patents_sample, data_source="patents"
    )
    publication_abstract_dict = patents.set_index("publication_number")[
        "abstract_localized"
    ].to_dict()
    patents_sample_df["abstract"] = patents_sample_df.id.map(publication_abstract_dict)

    upload_obj(
        patents_sample_df[
            [
                "id",
                "abstract",
                "data_source",
                "entity",
                "confidence_score",
                "spacy_label",
            ]
        ],
        BUCKET_NAME,
        VALIDATION_SAMPLE_PATH,
    )
