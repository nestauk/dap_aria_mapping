from typing import List
import yaml
import networkx as nx
from nesta_ds_utils.networks.build import build_coocc
from nesta_ds_utils.loading_saving.S3 import upload_obj
from dap_aria_mapping.getters.openalex import get_openalex_entities
from dap_aria_mapping.getters.patents import get_patent_entities
from dap_aria_mapping import BUCKET_NAME


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
            if entity["confidence"] >= confidence_threshold:
                abstract_list.append(entity["entity"])
        cooccurrence_data.append(abstract_list)

    return cooccurrence_data


if __name__ == "__main__":

    # load in taxonomy config
    with open("dap_aria_mapping/config/taxonomy.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

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

    # build term cooccurrence network
    print("Generating network")
    network = build_coocc(cooccurrence_data, edge_attributes=["association_strength"])

    # remove nodes with degree <= frequency threshold
    to_be_removed = [
        x
        for x in network.nodes()
        if network.degree(x, weight="weight") <= config["min_entity_frequency"]
    ]
    for x in to_be_removed:
        network.remove_node(x)
    print("Network generated with {} nodes".format(network.number_of_nodes()))

    # save network to S3
    print("Saving network as pickle file to S3")
    upload_obj(network, BUCKET_NAME, "outputs/cooccurrence_network.pkl")
