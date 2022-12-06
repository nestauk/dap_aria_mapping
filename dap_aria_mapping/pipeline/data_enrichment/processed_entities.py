"""
post-process entities pipeline
--------------
Pipeline to post process extracted entities using
spacy's pretrained NER model where expected input of extracted
entities includes both the DBpedia URI and confidence score.

Filter out entities that are: people, organisations, locations, money

For example,
entities = {'EP-3810804-A1': [{'URI': 'http://dbpedia.org/resource/Alternative_splicing',
   'confidence': 80},
  {'URI': 'http://dbpedia.org/resource/Genomics', 'confidence': 70},
  {'URI': 'http://dbpedia.org/resource/Transcriptome', 'confidence': 100},
  {'URI': 'http://dbpedia.org/resource/Protein', 'confidence': 90},
  {'URI': 'http://dbpedia.org/resource/RNA', 'confidence': 90},
  {'URI': 'http://dbpedia.org/resource/Machine_learning', 'confidence': 90}],
 'EP-1967857-A3': [],
 'US-2007134705-A1': [{'URI': 'http://dbpedia.org/resource/Transcription_factor',
   'confidence': 90},
  {'URI': 'http://dbpedia.org/resource/Gene', 'confidence': 90},
  {'URI': 'http://dbpedia.org/resource/Interval_graph', 'confidence': 100}],
 'US-8921074-B2': [{'URI': 'http://dbpedia.org/resource/Colorectal_cancer',
   'confidence': 100},
  {'URI': 'http://dbpedia.org/resource/Gene', 'confidence': 90},
  {'URI': 'http://dbpedia.org/resource/IL2RB', 'confidence': 90},
  {'URI': 'http://dbpedia.org/resource/TSG-6', 'confidence': 90},
  {'URI': 'http://dbpedia.org/resource/RNA', 'confidence': 90}],
 'WO-2020092591-A1': [{'URI': 'http://dbpedia.org/resource/Germline',
   'confidence': 70},
  {'URI': 'http://dbpedia.org/resource/Genome', 'confidence': 90},
  {'URI': 'http://dbpedia.org/resource/Allele', 'confidence': 90}]}
clean_entities = {'EP-3810804-A1': ['Alternative splicing', 'Protein', 'Machine learning'],
 'EP-1967857-A3': [],
 'US-2007134705-A1': ['Gene', 'Interval graph'],
 'US-8921074-B2': ['Colorectal cancer', 'Gene', 'TSG-6'],
 'WO-2020092591-A1': ['Germline', 'Genome']}
"""

from dap_aria_mapping import BUCKET_NAME

from metaflow import FlowSpec, step, retry
import ast
import boto3
import json
from typing import List, Dict
import spacy

client = boto3.client("s3")
S3 = boto3.resource("s3")

BAD_ENTS = ["ORG", "GPE", "MONEY", "LOC", "PERSON"]
# sam to output extracted entities in this s3 path
ENTITIES_PATH = "inputs/data_enrichment/extracted_entities/"
NER = spacy.load("en_core_web_sm")


def predict_entity_type(entities_list: List[str]) -> List[int]:
    """Predicts whether an entity type contains
    bad entites (==1) or not (==0) for a given entities list

    Returns a list of predictions.
    """
    labels = []
    for i, ent in enumerate(entities_list):
        bad_ents = []
        doc = NER(ent)
        if doc.ents:
            for tok in doc.ents:
                if tok.label_ in BAD_ENTS:
                    bad_ents.append(entities_list[i])
        bad_ents_deduped = list(set(bad_ents))
        if bad_ents_deduped != []:
            labels.append(1)
        else:
            labels.append(0)

    return labels


def filter_entities(
    entities_list: List[Dict[str, str]], confidence_threshold: int = 60
) -> List[str]:
    """Filters entities list based on list of predicted entity types

    Returns a list of entities predicted as not containing bad entities
        and above a confidence threshold.
    """
    ents = []
    for ent in entities_list:
        split_ent = ent["URI"].split("/")[-1].replace("_", " ").lower()
        if ent["confidence"] >= confidence_threshold:
            ents.append(split_ent)

    ent_preds = predict_entity_type([ent for ent in ents])

    return [ents[i] for i, pred in enumerate(ent_preds) if pred != 1]


def get_all_lookups():
    """Get all lookups stored on the S3 bucket 'aria-mapping'
    or in a specific directory.
    Returns:
        list: All lookups.
    """
    batches = [
        obj["Key"]
        for obj in client.list_objects(
            Bucket=BUCKET_NAME, Prefix=ENTITIES_PATH, Delimiter="/"
        )["Contents"]
        if obj["Key"].endswith("_lookup.json")
    ]
    return batches


class PostprocessEntitiesFlow(FlowSpec):
    @step
    def start(self):
        self.links = get_all_lookups()
        self.next(self.postprocess, foreach="links")

    @retry
    @step
    def postprocess(self):
        from nesta_ds_utils.loading_saving.S3 import upload_obj

        content_object = S3.Object(BUCKET_NAME, self.input)
        file_content = content_object.get()["Body"].read().decode("utf-8")
        entities = ast.literal_eval(json.loads(json.dumps(file_content)))
        clean_entities = {
            text_id: filter_entities(entity) for text_id, entity in entities.items()
        }
        filename = f"{self.input.replace('.json', '_clean.json')}"
        upload_obj(BUCKET_NAME, clean_entities, filename)
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    PostprocessEntitiesFlow()
