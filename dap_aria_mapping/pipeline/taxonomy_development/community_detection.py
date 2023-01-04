import yaml
from typing import List
import networkx as nx
from networkx.algorithms.community import louvain_communities
from nesta_ds_utils.loading_saving.S3 import download_obj, upload_obj
from dap_aria_mapping import BUCKET_NAME
from collections import defaultdict
import pandas as pd


def generate_hierarchy(
    node_list: List,
    network: nx.Graph,
    hierarchy: dict,
    resolution: int,
    n_iter: int = 1,
    resolution_increments: int = 1,
    min_group_size: int = 10,
    total_iters: int = 5,
) -> dict:
    """recursively splits lists of nodes into sub-communities

    Args:
        node_list (List): list of nodes to split
        network (nx.Graph): term co-occurrence network used to group nodes
        hierarchy (dict): community assignments at previous split, key: entity, value: community
        resolution (int): resolution to use for community detection
        n_iter (int, optional): current iteration of recursion. Defaults to 1.
        resolution_increments (int, optional): how much to increase resolution at each split. Defaults to 1.
        min_group_size (int, optional): do not split node list if it is smaller than min group size. Defaults to 10.
        total_iters (int, optional): max number of splits for entire algorithm. Defaults to 5.

    Returns:
        dict: key: entity, value: community assignment at each level.
            Community assignments are expressed as <LEVEL1>_<LEVEL2>_<LEVEL3>
    """

    while n_iter < total_iters and len(node_list) > min_group_size:
        subgraph = network.subgraph(node_list)
        communities = louvain_communities(
            subgraph, resolution=resolution, seed=1, weight="association_strength"
        )
        numbered_communities = enumerate(communities)
        for index, node_list in numbered_communities:
            for node in node_list:
                node_upper_level_community = hierarchy[node]
                node_next_level_community = node_upper_level_community + "_{}".format(
                    str(index)
                )
                hierarchy[node] = node_next_level_community
            hierarchy = generate_hierarchy(
                node_list,
                network,
                hierarchy,
                resolution + resolution_increments,
                n_iter + 1,
            )
    return hierarchy


def format_output(hierarchy: dict, total_splits: int) -> pd.DataFrame:
    """formats the clustering hierarchy into table

    Args:
        hierarchy (dict): clustering hierarchy
        total_splits (int): total number of levels in the hierarchy

    Returns:
        pd.DataFrame: table where index is entity name and columns are cluster assignments at each level
    """
    output_list = []
    for entity, community in hierarchy.items():
        temp = [None] * (total_splits + 1)
        temp[0] = entity
        split_communs = community.split("_")
        n_splits = len(split_communs)
        for i in range(1, total_splits + 1):
            commun_str = "_".join(split_communs[:i])
            try:
                temp[i] = commun_str
            except IndexError:
                print("Error with entity {} in community {}".format(entity, community))
                continue
        output_list.append(temp)
    output_columns = ["Entity"]
    for i in range(1, total_splits + 1):
        output_columns.append("Level_{}".format(i))
    output_df = pd.DataFrame(output_list)

    output_df.columns = output_columns
    output_df.set_index("Entity", inplace=True)

    return output_df


if __name__ == "__main__":

    # load pre-computed cooccurrence network from S3
    print("Loading network")
    network = download_obj(BUCKET_NAME, "outputs/cooccurrence_network.pkl")

    # load taxonomy config file with parameters
    with open("dap_aria_mapping/config/taxonomy.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)["community_detection"]

    # recursively split network into communities
    print("Splitting network into communities")
    starting_resolution = config["starting_resolution"]
    initial_partitions = louvain_communities(
        network, resolution=starting_resolution, seed=1, weight="association_strength"
    )
    numbered_communities = enumerate(initial_partitions)

    hierarchy = defaultdict(str)
    for index, node_list in numbered_communities:
        for node in node_list:
            hierarchy[node] = str(index)

        hierarchy = generate_hierarchy(
            node_list,
            network,
            hierarchy,
            resolution=starting_resolution + config["resolution_increments"],
            resolution_increments=config["resolution_increments"],
            min_group_size=config["min_group_size"],
            total_iters=config["max_splits"],
        )

    # format output as table and save as parquet to S3
    print("Formatting output")
    output = format_output(hierarchy, config["max_splits"])
    print(output.head())

    print("Saving output")
    upload_obj(
        output,
        BUCKET_NAME,
        "outputs/community_detection_clusters.parquet",
        kwargs_writing={"index": True},
    )
