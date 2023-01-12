import yaml
from typing import List, Union
import networkx as nx
from networkx.algorithms.community import louvain_communities
from nesta_ds_utils.loading_saving.S3 import upload_obj
from dap_aria_mapping import BUCKET_NAME
from collections import defaultdict
import pandas as pd
from dap_aria_mapping.getters.cooccurrence_network import (
    get_cooccurrence_network,
    get_test_cooccurrence_network,
)
from dap_aria_mapping.getters.taxonomies import get_taxonomy_config
import argparse


def generate_hierarchy(
    node_list: List,
    network: nx.Graph,
    hierarchy: dict,
    resolution: int,
    n_iter: int = 1,
    resolution_increments: int = 1,
    min_group_size: int = 10,
    total_iters: int = 5,
    seed: Union[int,None] = 1
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
        seed (Union[int,None]): seed to use for community detection, if None will result in different results each run.

    Returns:
        dict: key: entity, value: community assignment at each level.
            Community assignments are expressed as <LEVEL1>_<LEVEL2>_<LEVEL3>
    """

    while n_iter < total_iters and len(node_list) > min_group_size:
        subgraph = network.subgraph(node_list)
        communities = louvain_communities(
            subgraph, resolution=resolution, seed=seed, weight="association_strength"
        )
        for index, node_list in enumerate(communities):
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
    # parse community string and format as list
    output_list = []
    for entity, community in hierarchy.items():
        temp = [None] * (total_splits + 1)
        temp[0] = entity
        split_communs = community.split("_")
        for i in range(1, total_splits + 1):
            commun_str = "_".join(split_communs[:i])
            temp[i] = commun_str
        output_list.append(temp)

    # build dataframe and set column names and index
    output_columns = ["Entity"]
    for i in range(1, total_splits + 1):
        output_columns.append("Level_{}".format(i))
    output_df = pd.DataFrame(output_list)
    output_df.columns = output_columns
    output_df.set_index("Entity", inplace=True)

    return output_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test Mode", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="run script in test mode on small sample",
    )
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        help="save output locally rathern than to S3",
    )
    args = parser.parse_args()

    # load pre-computed cooccurrence network from S3
    print("Loading network")
    if args.local:
        if args.test:
            network = get_test_cooccurrence_network(local=True)
        else:
            network = get_cooccurrence_network(local=True)
    else:
        if args.test:
            network = get_test_cooccurrence_network()
        else:
            network = get_cooccurrence_network()

    # load taxonomy config file with parameters for community detection
    config = get_taxonomy_config()["community_detection"]

    # recursively split network into communities
    print("Splitting network into communities")
    starting_resolution = config["starting_resolution"]
    initial_partitions = louvain_communities(
        network, resolution=starting_resolution, seed=config["seed"], weight="association_strength"
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
    if args.local:
        if args.test:
            output.to_parquet("outputs/test_community_detection_clusters.parquet", index=True)
        else:
            output.to_parquet("outputs/community_detection_clusters.parquet", index=True)

    
    else:
        if args.test:
            upload_obj(
                output,
                BUCKET_NAME,
                "outputs/test_community_detection_clusters.parquet",
                kwargs_writing={"index": True},
            )
        else:
            upload_obj(
                output,
                BUCKET_NAME,
                "outputs/community_detection_clusters.parquet",
                kwargs_writing={"index": True},
            )
