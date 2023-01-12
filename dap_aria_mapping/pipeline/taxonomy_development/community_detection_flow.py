import os

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements_community_detection.txt 1> /dev/null"
)

from typing import List
import networkx as nx
from networkx.algorithms.community import louvain_communities
from nesta_ds_utils.loading_saving.S3 import download_obj, upload_obj
from collections import defaultdict
import pandas as pd

from metaflow import FlowSpec, step, Parameter, retry, batch

BUCKET_NAME = "aria-mapping"

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


class CommunityDetectionFlow(FlowSpec):
    min_group_size = Parameter(
        "min_group_size",
        default=10,
        help="minimum number of nodes in a community to split",
    )
    max_splits = Parameter(
        "max_splits",
        default=5,
        help="maximum number of splits to perform on a community",
    )
    starting_resolution = Parameter(
        "starting_resolution",
        default=1,
        help="resolution to use for first split",
    )
    resolution_increments = Parameter(
        "resolution_increments",
        default=1,
        help="how much to increase resolution at each split",
    )

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.load_data)

    @step
    def load_data(self):
        """loads pre-computed cooccurrence network and config file"""
        # load pre-computed cooccurrence network from S3
        print("Loading network")
        self.network = download_obj(BUCKET_NAME, "outputs/cooccurrence_network.pkl")
        self.next(self.generate_initial_partitions)

    @retry
    @batch(cpu=2, memory=48000)
    @step
    def generate_initial_partitions(self):
        """splits network into initial set of partitions"""
        print("Splitting network into communities")
        self.initial_partitions = louvain_communities(
            self.network,
            resolution=self.starting_resolution,
            seed=1,
            weight="association_strength",
        )
        self.next(self.split_partitions)

    @retry
    @batch(cpu=2, memory=48000)
    @step
    def split_partitions(self):
        """recursively split initial partitions to generate hierarchy"""
        self.numbered_communities = enumerate(self.initial_partitions)
        self.hierarchy = defaultdict(str)
        for index, node_list in self.numbered_communities:
            for node in node_list:
                self.hierarchy[node] = str(index)

            self.hierarchy = generate_hierarchy(
                node_list,
                self.network,
                self.hierarchy,
                resolution=self.starting_resolution
                + self.resolution_increments,
                resolution_increments=self.resolution_increments,
                min_group_size=self.min_group_size,
                total_iters=self.max_splits,

            )

        self.next(self.format_and_save)

    @retry
    @batch(cpu=2, memory=48000)
    @step
    def format_and_save(self):
        """format hierarchy into table and save to S3 as parquet file"""
        output = format_output(self.hierarchy, self.max_splits)
        upload_obj(
            output,
            BUCKET_NAME,
            "outputs/test_community_detection_clusters.parquet",
            kwargs_writing={"index": True},
        )
        self.next(self.end)

    @step
    def end(self):
        """Ends the flow"""
        pass


if __name__ == "__main__":
    CommunityDetectionFlow()
