from typing import List
import yaml
import networkx as nx
from nesta_ds_utils.networks.build import build_coocc
from nesta_ds_utils.loading_saving.S3 import upload_obj
from dap_aria_mapping.getters.openalex import get_openalex_entities
from dap_aria_mapping.getters.patents import get_patent_entities
from dap_aria_mapping.getters.taxonomies import get_taxonomy_config
from dap_aria_mapping import BUCKET_NAME
import argparse
import pickle

def generate_cooccurrence_data(
    raw_entity_data: List[dict],
    confidence_threshold: int = 80,
    cooccurrence_data: List[list] = [],
) -> List[list]:
    """generates list of list of entities contained in each abstract

    Args:
        raw_entity_data (List[dict]):
        confidence_threshold (int, optional): only include entities with confidence >= threshold. Defaults to 80.
        cooccurrence_data (List[list], optional): output data to append to. Defaults to [].

    Returns:
        List: list of list of entities contained in each abstract
    """
    entity_list = [item for item in raw_entity_data.values() if len(item) > 0]

    for abstract in entity_list:
        abstract_list = []
        for entity in abstract:
            if (entity["confidence"] >= confidence_threshold):
                abstract_list.append(entity["entity"])
        cooccurrence_data.append(abstract_list)

    return cooccurrence_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test Mode", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-t",
        "--test_mode",
        action="store_true",
        help="run script in test mode on small sample",
    )
    parser.add_argument(
        "-l",
        "--local_output",
        action="store_true",
        help="save output locally rathern than to S3",
    )
    args = parser.parse_args()

    # load in taxonomy config
    config = get_taxonomy_config()

    # load in raw entity data
    print("Loading raw data")
    oa_entities = get_openalex_entities()
    patent_entities = get_patent_entities()

    # build combined list of lists of entities in each abstract
    print("Building input data")
    cooccurrence_data = generate_cooccurrence_data(
        oa_entities, confidence_threshold=config["min_entity_confidence"]
    )
    cooccurrence_data = generate_cooccurrence_data(
        patent_entities,
        confidence_threshold=config["min_entity_confidence"],
        cooccurrence_data=cooccurrence_data,
    )
    # if running in test mode, only use a small sample of the data
    if args.test_mode:
        cooccurrence_data = cooccurrence_data[:1000]

    # build term cooccurrence network
    print("Generating network")
    network = build_coocc(cooccurrence_data, edge_attributes=["association_strength"], use_node_weights=True)

    # remove nodes with degree <= frequency threshold
    node_frequences = nx.get_node_attributes(network, "frequency")
    
    to_be_removed = [
        x
        for x in network.nodes()
        if node_frequences[x] < config["min_entity_frequency"]
    ]
    for x in to_be_removed:
        network.remove_node(x)
    
    print("Network generated with {} nodes".format(network.number_of_nodes()))

    # save network as pickle file
    print("Saving network as pickle file")
    if args.local_output:
        print("Saving locally")
        if args.test_mode:
            with open("outputs/test_cooccurrence_network.pkl","wb") as f:
                pickle.dump(network,f)
        else:
            with open("outputs/cooccurrence_network.pkl","wb") as f:
                pickle.dump(network,f)

    else:
        print("Saving to S3")
        if args.test_mode:
            upload_obj(network, BUCKET_NAME, "outputs/test_cooccurrence_network.pkl")
        else:
            upload_obj(network, BUCKET_NAME, "outputs/cooccurrence_network.pkl")
