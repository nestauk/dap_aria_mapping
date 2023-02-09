from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME
import networkx as nx
import pickle


def get_test_cooccurrence_network(local) -> nx.Graph():
    """gets small network of term cooccurrences generated from OpenAlex and Patents Data.
    Script used to generate network can be found in pipeline/taxonomy_development/build_cooccurrence_network.py
    run in test_mode.
    Parameters of network can be found in config/taxonomy.yaml

    Returns:
        nx.Graph: networkx graph of term cooccurrences
    """
    return download_obj(BUCKET_NAME, "outputs/test_cooccurrence_network.pkl")


def get_cooccurrence_network(local: bool = False) -> nx.Graph():
    """gets network of term cooccurrences generated from OpenAlex and Patents Data.
    Script used to generate network can be found in pipeline/taxonomy_development/build_cooccurrence_network.py.
    Parameters of network can be found in config/taxonomy.yaml
    Args:
        local (bool, optional): whether to pull locally or from S3, defaults to S3
    Returns:
        nx.Graph: networkx graph of term cooccurrences
    """
    if local:
        with open("outputs/cooccurrence_network.pkl", "rb") as f:
            network = pickle.load(f)
        return network
    else:
        return download_obj(BUCKET_NAME, "outputs/cooccurrence_network.pkl")
