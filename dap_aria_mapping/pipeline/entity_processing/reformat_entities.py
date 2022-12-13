from typing import Dict
import boto3
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
    if "dbpedia_entities" in dictionary.keys():
        dictionary["dbpedia_entities"] = [
            {
                "confidence": entity["confidence"],
                "URI": entity["URI"],
            }
            for entity in dictionary["dbpedia_entities"]
            if entity["confidence"] > 40
        ]
        return {
            "id": dictionary["id"],
            "dbpedia_entities": dictionary["dbpedia_entities"]
        }
    else:
        return {
            "id": dictionary["id"],
            "dbpedia_entities": []
        }


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

    @batch(cpu=2, memory=64000)
    @step
    def reformat_entities(self):
        """
        Reformat the entities dictionary to remove unnecessary information.
        """
        s3 = boto3.resource("s3")
        content_object = s3.Object("aria-mapping", self.input)
        file_content = content_object.get()['Body'].read().decode("utf-8")
        json_content = json.loads(file_content)
        reformatted_entities = [
            reformat_dictionary(item)
            for item in json_content
        ]
        if "annotated_patents.json" in self.input:
            s3object = s3.Object("aria-mapping", PATENT_ANNOTATION_OUTPUT)
        else:
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
