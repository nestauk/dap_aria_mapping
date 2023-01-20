from collections import defaultdict
import pandas as pd

def topic_distributions(docs_with_topics: dict) -> tuple(dict, dict):
    """takes a set of documents tagged with topics and returns 
        the distribution of the count of documents per topic

    Args:
        docs_with_topics (dict): key: document id, value: list of topics associated with the document

    Returns:
        tuple: returns two dictionaries representing distributions:
            1. count of documents per topic - key: topic id, value: count of documents
            2. proportion of documents per topic - key: topic id, value: percent of topics containing that topic
                note: sum of proportion distribution will not = 1 as a document can be assigned to multiple topics
    """
    absolute_dist = defaultdict(int)
    for doc, topics in docs_with_topics.items():
        for topic in set(topics):
            absolute_dist[topic] += 1

    proportional_dist = defaultdict(float)
    for key, val in absolute_dist.items():
        proportional_dist[key] = val/len(docs_with_topics)
        
    return absolute_dist, proportional_dist

def taxonomy_distribution(tax_at_level: pd.Series) -> dict:
    """takes the taxonomy at a given level and returns the distribution 
        of entities per category in the taxonomy 

    Args:
        tax_at_level (pd.Series): slice of the taxonomy dataframe at a given level

    Returns:
        dict: taxonomy distribution - key: taxonomy id, value: proportion of entities in the category
    """
    tax_dist = defaultdict(float)
    for taxid, count in pd.DataFrame(tax_dist.value_counts()).iterrows():
        tax_dist[taxid] = count[0]/len(tax_dist)
    return tax_dist

def topics_per_institution_distribution(docs_with_topics: dict, institution_lookup: dict) -> tuple(dict, dict):
    """takes documents tagged with topics and a lookup of documents to institutions and generates 
        a distribution of topics per institutions

    Args:
        docs_with_topics (dict):  key: document id, value: list of topics associated with the document
        institution_lookup (dict): key: document id, value: list of institutions associated with the document

    Returns:
         tuple: returns two dictionaries representing distributions:
            1. count of documents per institution per topic - 
                key: topic id, 
                value: dictionary - key: institution: value: count of documents
            2. proportion of documents per topic - 
                key: topic id, 
                value: dictionary - key: institution: value: percent of institutions
                note: sum of proportion distribution will not = 1 as an institution can produce 
                many documents which can have multiple topics

    """
    
    total_institutions = len(list(institution_lookup.keys()))
    absolute_dist = defaultdict(lambda: defaultdict(int))
    for doc, topics in docs_with_topics.items():
        inst_list = institution_lookup[doc]
        for institution in inst_list:
            for topic in set(topics):
                absolute_dist[topic][institution] += 1

    proportional_dist = defaultdict(lambda: defaultdict(int))
    for topic, institution_dict in absolute_dist.items():
        for inst, count in institution_dict.items():
            proportional_dist[topic][inst] = count/total_institutions
    
    return absolute_dist, proportional_dist