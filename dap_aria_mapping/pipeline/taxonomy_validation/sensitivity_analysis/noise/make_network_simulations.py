from typing import List, Sequence, Union
import yaml
import networkx as nx
from networkx.algorithms.community import louvain_communities
import numpy as np
import pandas as pd
from nesta_ds_utils.networks.build import build_coocc
from nesta_ds_utils.loading_saving.S3 import upload_obj
from dap_aria_mapping.getters.openalex import get_openalex_entities
from dap_aria_mapping.getters.patents import get_patent_entities

# from dap_aria_mapping.getters.taxonomies import get_taxonomy_taxonomy
from dap_aria_mapping import PROJECT_DIR, BUCKET_NAME, logger, taxonomy
import argparse
import pickle
from itertools import chain
from collections import Counter, defaultdict
from toolz import pipe
import random


def generate_cooccurrence_data(
    raw_entity_data: List[dict],
    min_confidence_threshold: int = 80,
    cooccurrence_data: Sequence = [],
) -> List[list]:
    """generates list of list of entities contained in each abstract

    Args:
        raw_entity_data (List[dict]): list of entities contained in each abstract, list items represent abstracts,
            dict: {"entity": value of entity, "confidence": confidence score from DBPedia entity extraction}
        min_confidence_threshold (int, optional): only include entities with confidence >= threshold. Defaults to 80.
        cooccurrence_data (Sequence, optional): output data to append to. Defaults to [].

    Returns:
        List: list of list of entities contained in each abstract
    """
    entity_list = [item for item in raw_entity_data.values() if len(item) > 0]

    for abstract in entity_list:
        abstract_list = []
        for entity in abstract:
            if entity["confidence"] >= min_confidence_threshold:
                abstract_list.append(entity["entity"])
        cooccurrence_data.append(abstract_list)

    return cooccurrence_data


def generate_additional_cooccurrence_data(
    raw_entity_data: List[dict],
    min_confidence_threshold: int = 80,
    cooccurrence_data: Sequence = [],
) -> List[list]:
    entity_list = [item for item in raw_entity_data.values() if len(item) > 0]

    for abstract in entity_list:
        abstract_list = []
        for entity in abstract:
            if entity["confidence"] < min_confidence_threshold:
                abstract_list.append(entity["entity"])
        cooccurrence_data.append(abstract_list)

    return cooccurrence_data


def generate_hierarchy(
    node_list: List,
    network: nx.Graph,
    hierarchy: dict,
    resolution: int,
    n_iter: int = 1,
    resolution_increments: int = 1,
    min_group_size: int = 10,
    total_iters: int = 5,
    seed: Union[int, None] = 1,
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
        "--local_output",
        action="store_true",
        help="save output locally rathern than to S3",
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=1,
        help="Number of simulations to run",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load config
    logger.info("Loading config")
    config = taxonomy["community_detection"]

    # load in raw entity data
    logger.info("Loading raw data")
    oa_entities = get_openalex_entities()
    patent_entities = get_patent_entities()

    # build combined list of lists of entities in each abstract
    logger.info("Building input data")
    cooccurrence_data = generate_cooccurrence_data(
        oa_entities, min_confidence_threshold=config["min_entity_confidence"]
    )
    cooccurrence_data = generate_cooccurrence_data(
        patent_entities,
        min_confidence_threshold=config["min_entity_confidence"],
        cooccurrence_data=cooccurrence_data,
    )

    cooccurrence_data_discarded = generate_additional_cooccurrence_data(
        oa_entities, min_confidence_threshold=config["min_entity_confidence"]
    )
    cooccurrence_data_discarded = generate_additional_cooccurrence_data(
        patent_entities,
        min_confidence_threshold=config["min_entity_confidence"],
        cooccurrence_data=cooccurrence_data_discarded,
    )

    del oa_entities, patent_entities

    logger.info("Filtering entities")
    entity_counts = Counter(chain(*cooccurrence_data))
    entities = set(
        [
            x
            for x in entity_counts
            if entity_counts[x] >= config["min_entity_frequency"]
        ]
    )
    logger.info(f"Number of unique entities in taxonomy = {len(entities)}")

    entities_discarded = pipe(
        cooccurrence_data_discarded,
        lambda x: set(chain(*x)),
        lambda x: x - entities,
    )

    logger.info(f"Number of unique additional entities = {len(entities_discarded)}")

    cooccurrence_data_clean = [
        [x for x in ls if x in entities] for ls in cooccurrence_data
    ]
    cooccurrence_data_discarded_clean = [
        [x for x in ls if x in entities_discarded] for ls in cooccurrence_data_discarded
    ]

    ratio = len(entities_discarded) / len(entities)

    for percentage in np.logspace(-1, 0, 10):
        ratio_str = str(round(percentage * ratio, 2))[-2:]
        logger.info(
            "Noise level: " + str(round(percentage, 2)) + " - ratio: " + ratio_str
        )

        for simul in range(args.n_simulations):
            entities_sample = random.sample(
                entities_discarded, int(percentage * len(entities_discarded))
            )
            cooccurrence_data_discarded_sample = [
                [x for x in ls if x in entities_sample]
                for ls in cooccurrence_data_discarded_clean
            ]

            cooccurrence_data_sample = list(
                chain(*zip(cooccurrence_data_clean, cooccurrence_data_discarded_sample))
            )

            # build term cooccurrence network
            logger.info("Generating network")
            network = build_coocc(
                cooccurrence_data_sample,
                edge_attributes=["association_strength"],
                use_node_weights=True,
            )

            logger.info(
                "Network generated with {} nodes".format(network.number_of_nodes())
            )

            # save network as pickle file
            logger.info("Saving network as pickle file")
            if args.local_output:
                OUTPUT_PATH = (
                    PROJECT_DIR / "outputs" / "simulations" / "noise" / "raw_outputs"
                )
                OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
                logger.info("Saving locally")
                if args.test:
                    with open(
                        OUTPUT_PATH
                        / f"test_cooccurrence_network_ratio_{ratio}_simul_{simul}.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(network, f)
                else:
                    with open(
                        OUTPUT_PATH
                        / f"cooccurrence_network_ratio_{ratio}_simul_{simul}.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(network, f)

            else:
                logger.info("Saving to S3")
                if args.test:
                    upload_obj(
                        network,
                        BUCKET_NAME,
                        f"outputs/validation_metrics/simulations/noise/raw_outputs/test_cooccurrence_network_ratio_{ratio}_simul_{simul}.pkl",
                    )
                else:
                    upload_obj(
                        network,
                        BUCKET_NAME,
                        f"outputs/validation_metrics/simulations/noise/raw_outputs/cooccurrence_network_ratio_{ratio}_simul_{simul}.pkl",
                    )

            # run Louvain community detection
            logger.info("Splitting network into communities")
            starting_resolution = config["starting_resolution"]
            initial_partitions = louvain_communities(
                network,
                resolution=starting_resolution,
                seed=args.seed,
                weight="association_strength",
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
            logger.info("Formatting output")
            output = format_output(hierarchy, config["max_splits"])
            print(output.head())

            logger.info("Saving output")
            if args.local_output:
                OUTPUT_PATH = (
                    PROJECT_DIR
                    / "outputs"
                    / "simulations"
                    / "noise"
                    / "formatted_outputs"
                )
                logger.info("Saving locally")
                if args.test:
                    with open(
                        OUTPUT_PATH
                        / f"test_community_detection_clusters_{ratio}_simul_{simul}.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(network, f)
                else:
                    with open(
                        OUTPUT_PATH
                        / f"community_detection_clusters_{ratio}_simul_{simul}.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(network, f)

            else:
                logger.info("Saving to S3")
                if args.test:
                    upload_obj(
                        network,
                        BUCKET_NAME,
                        f"outputs/validation_metrics/simulations/noise/formatted_outputs/test_community_detection_clusters_ratio_{ratio}_simul_{simul}.pkl",
                    )
                else:
                    upload_obj(
                        network,
                        BUCKET_NAME,
                        f"outputs/validation_metrics/simulations/noise/formatted_outputs/community_detection_clusters_{ratio}_simul_{simul}.pkl",
                    )
