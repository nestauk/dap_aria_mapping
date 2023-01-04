from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME
import networkx as nx


def get_cooccurrence_network() -> nx.Graph():
    """gets network of term cooccurrences generated from OpenAlex and Patents Data.
    Script used to generate network can be found in pipeline/taxonomy_development/build_cooccurrence_network.py.
    Parameters of network can be found in config/taxonomy.yaml

    Returns:
        nx.Graph: networkx graph of term cooccurrences
    """
    return download_obj(BUCKET_NAME, "outputs/cooccurrence_network.pkl")
