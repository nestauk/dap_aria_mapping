from dap_aria_mapping.getters.taxonomies import get_cooccurrence_taxonomy, get_cooccurrence_taxonomy
from dap_aria_mapping.getters.openalex import get_openalex_entities
from dap_aria_mapping.getters.patents import get_patent_entities
from collections import defaultdict
from dap_aria_mapping import BUCKET_NAME
from nesta_ds_utils.loading_saving.S3 import upload_obj
import pandas as pd
import json

def tag_docs_at_level(docs: dict, tax_at_level: pd.Series) -> dict:
    """tag documents with topic labels based on entities present in document

    Args:
        docs (dict): dictionary where key: document id, value: list of entity information about each document
        tax_at_level (pd.Series): taxonomy of entities, index: entity, value: topic of entity at a given level of the taxonomy

    Returns:
        dict: key: documentid, value: list of topics per document at given level of taxonomy 
    """
    topics_per_doc = defaultdict()
    for docid, entity_list in docs.items():
        doc_holder = []
        for entity_info in entity_list:
            try:
                if entity_info["confidence"]>=80:
                    topic = tax_at_level.loc[entity_info["entity"]]
                    doc_holder.append(topic)
            except KeyError:
                continue
        topics_per_doc[docid] = doc_holder
    return topics_per_doc

if __name__ == '__main__':

    print("loading taxonomy")
    taxonomy = get_cooccurrence_taxonomy()
    """
    print("TAGGING PATENTS WITH TOPICS")
    print("loading patents data")
    patents = get_patent_entities()
    
    for level in range(2,len(taxonomy.columns)+1):
        print("starting tagging for level {}".format(level))
        tax_at_level = taxonomy["Level_{}".format(level)]
        topics_per_doc = tag_docs_at_level(patents, tax_at_level)
        upload_obj(
            topics_per_doc,
            BUCKET_NAME,
            'outputs/docs_with_topics/patents/Level_{}.json'.format(level)
        )
    """

    print("TAGGING OPENALEX WITH TOPICS")
    print("loading openalex data")
    openalex = get_openalex_entities()
    for level in range(1,len(taxonomy.columns)+1):
        print("starting tagging for level {}".format(level))
        tax_at_level = taxonomy["Level_{}".format(level)]
        topics_per_doc = tag_docs_at_level(openalex, tax_at_level)
        #tax_df = pd.DataFrame.from_dict(topics_per_doc, orient = 'index')
        #columns = ["Topic_{}".format(i) for i in range(0,len(tax_df.columns))]
        #tax_df.columns = columns
        with open('dap_aria_mapping/pipeline/topic_distributions/outputs/openalex/Level_{}.json'.format(level), 'w+') as f:
            json.dump(topics_per_doc,f)


