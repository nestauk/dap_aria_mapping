import os

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/reformat_entities_requirements.txt 1> /dev/null"
)

from typing import Dict
import boto3
import pandas as pd
import json

from metaflow import FlowSpec, step, Parameter, batch


PATENT_ANNOTATION_OUTPUT = "inputs/data_collection/patents/preprocessed_annotated_patents.json"
OPENALEX_ANNOTATION_OUTPUT = "inputs/data_collection/processed_openalex/preprocessed_annotated_abstracts.json"
INPUTS = [
    "inputs/data_collection/patents/annotated_patents.json",
    "inputs/data_collection/processed_openalex/annotated_abstracts.json"
]


def reformat_dictionary(dictionary: Dict) -> Dict:
    """
    Cleans out the unnecessary information from the annotation output dictionary
    """
    clean_dictionary = dict()
    for id_, ents in dictionary.items():
        clean_ents = []
        for e in ents:
            if e['confidence'] >= 70:
                clean_ents.append({'entity':e['URI'].split('/')[-1].replace('_', ' '),
                                    'confidence': e['confidence']})
        if clean_ents != []:
            clean_dictionary[id_] = clean_ents
        else:
            clean_dictionary[id_] = []
    return clean_dictionary


class EntityReformattingFlow(FlowSpec):
    production = Parameter("production", help="Run in production?", default=False)
    @step
    def start(self):
        """
        Starts the flow.
        """
        if self.production:
            self.input_files = INPUTS
        else:
            self.input_files = INPUTS[:1]
        self.next(self.reformat_entities, foreach="input_files")

    @batch(cpu=8, memory=200000)
    @step
    def reformat_entities(self):
        """
        Reformat the entities dictionary to remove unnecessary information.
        """
        s3 = boto3.resource("s3")
        content_object = s3.Object("aria-mapping", self.input)
        file_content = content_object.get()['Body'].read().decode("utf-8")
        json_content = json.loads(file_content)
        reformatted_entities = reformat_dictionary(json_content)
        if "annotated_patents.json" in self.input:
            s3object = s3.Object("aria-mapping", PATENT_ANNOTATION_OUTPUT)
        elif "annotated_abstracts.json" in self.input:
            s3object = s3.Object("aria-mapping", OPENALEX_ANNOTATION_OUTPUT)
        s3object.put(
            Body=(bytes(json.dumps(reformatted_entities).encode("UTF-8")))
        )
        self.next(self.dummy_join)

    @step
    def dummy_join(self, inputs):
        """Dummy join step"""
        self.next(self.end)

    @step
    def end(self):
        """Ends the flow"""
        pass

if __name__ == "__main__":
    EntityReformattingFlow()
