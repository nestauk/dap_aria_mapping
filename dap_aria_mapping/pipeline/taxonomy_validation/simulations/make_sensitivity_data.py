import re, argparse, boto3, json
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from toolz import pipe
from collections import defaultdict
from itertools import chain, combinations, compress
from typing import Dict, Sequence, Tuple
from dap_aria_mapping.getters.simulations import (
    get_simulation_outputs,
    get_simulation_topic_names,
)
from dap_aria_mapping.getters.taxonomies import get_entity_embeddings
from dap_aria_mapping.getters.validation import (
    get_topic_groups,
    get_subtopic_groups,
)
from dap_aria_mapping import logger, PROJECT_DIR, taxonomy, BUCKET_NAME
from nesta_ds_utils.loading_saving.S3 import upload_obj


# Topic sizes
def get_topic_sizes(
    dataframe: pd.DataFrame, taxonomy_class: str, simulation: str, ratio: str
) -> pd.DataFrame:
    """Iterates through levels and gets topic sizes for each level. The size+
        is the number of entities in the topic, divided by the total number of
        entities in the simulation.

    Args:
        dataframe (pd.DataFrame): A dataframe with the simulation outputs.
        taxonomy_class (str): A string indicating the taxonomy class. Must be
            one of "community", "centroids", "imbalanced".
        simulation (str): A string indicating the simulation family. Must be
            one of "noise", "sample".
        ratio (str): A string indicating the ratio of entities to topics. If
            simulation is "noise", must be one of "00", "05", "07", "08", "11",
            "14", "18", "24", "31", "39", "51". If simulation is "sample", must
            be one of "00", "10", "20", "30", "40", "50", "60", "70", "80", "90".

    Raises:
        ValueError: If no single key is found in the dictionary.

    Returns:
        pd.DataFrame: A dataframe with the topic sizes for each level.
    """
    ls_df = [
        get_topic_size(
            dataframe,
            taxonomy_class=taxonomy_class,
            level=level,
            ratio=ratio,
            simulation=simulation,
        )
        for level in range(1, 6)
    ]
    return pd.concat(ls_df).reset_index(drop=True)


def get_topic_size(
    df: pd.DataFrame,
    taxonomy_class: str,
    level: int = 1,
    simulation: str = "noise",
    ratio: str = "07",
) -> pd.DataFrame:
    """Gets the topic sizes for a given level.

    Args:
        df (pd.DataFrame): A dataframe with the simulation outputs.
        taxonomy_class (str): A string indicating the taxonomy class. Must be
            one of "community", "centroids", "imbalanced".
        level (int, optional): The level of the taxonomy. Must be one of 1, 2, 3,
            4, 5. Defaults to 1.
        simulation (str, optional): A string indicating the simulation family. Must be
            one of "noise", "sample". Defaults to "noise".
        ratio (str, optional): A string indicating the ratio of entities to topics. If
            simulation is "noise", must be one of "00", "05", "07", "08", "11",
            "14", "18", "24", "31", "39", "51". If simulation is "sample", must
            be one of "00", "10", "20", "30", "40", "50", "60", "70", "80", "90".

    Returns:
        pd.DataFrame: A dataframe with the topic sizes for the given level.
    """
    df_len = len(df)
    dict_topic_names = get_simulation_topic_names(
        taxonomy_class=taxonomy_class, level=level, simulation=simulation, ratio=ratio
    )
    return (
        df.groupby(f"Level_{str(level)}")
        .size()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={0: "size", f"Level_{str(level)}": "Topic"})
        .assign(
            PropEntities=lambda df: (df["size"] / df_len)
            + np.random.uniform(-5e-06, 5e-06, len(df)),
            Proportion=lambda df: round(df["PropEntities"], 5),
            level=f"Level_{str(level)}",
            parent_topic=lambda df: df["Topic"].str.split("_").str[0],
            TopicName=lambda df: df["Topic"].map(dict_topic_names),
        )
    )


# Centroid distances
def get_centroid_distance(
    dfgroup: DataFrameGroupBy, embeddings: Dict[str, np.ndarray], level: int = 1
) -> pd.Series:
    """Get the distances of each entity from the centroid of a given topic. Only entities
        in the actual taxonomy are included in the output.

    Args:
        dfgroup (DataFrameGroupBy): A groupby object with the entities in a given topic.
        embeddings (Dict[str, np.ndarray]): A dictionary with the entity embeddings.
        level (int, optional): The level of the taxonomy. Must be one of 1, 2, 3,
            4, 5. Defaults to 1.

    Returns:
        pd.Series: A series with the distances of each entity from the centroid of the topic.
    """
    if len(dfgroup) > 1:
        group_embeddings = np.array([embeddings[x] for x in dfgroup.index])
        group_centroid = np.mean(group_embeddings, axis=0)
        distances = np.linalg.norm(group_embeddings - group_centroid, axis=1)
        dfgroup[f"Level_{str(level)}_dist"] = distances
    else:
        dfgroup[f"Level_{str(level)}_dist"] = np.nan
    return dfgroup[f"Level_{str(level)}_dist"]


def get_centroid_distances(
    df: pd.DataFrame, embeddings: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Build a dataset with the distances of each entity from the centroid of each topic.

    Args:
        df (pd.DataFrame): A dataframe with the simulation outputs.
        embeddings (Dict[str, np.ndarray]): A dictionary with the entity embeddings.

    Returns:
        pd.DataFrame: A dataframe with the distances of each entity from the centroid of each topic.
    """
    cols = [
        pipe(
            df,
            lambda df: df.groupby(f"Level_{str(level)}"),
            lambda group: group.apply(
                lambda group: get_centroid_distance(group, embeddings, level)
            ),
            lambda df: df.reset_index().set_index("Entity")[
                [f"Level_{str(level)}_dist"]
            ],
            lambda df: df.loc[list(set(df.index).intersection(ENTITIES))],
        )
        for level in range(1, 6)
    ]
    distances = pd.concat(cols, axis=1)
    return distances


# Survival analysis
def generate_pairwise_topic_depths(
    taxonomy: pd.DataFrame,
    entity_groups: Dict[str, Dict[str, Sequence[str]]],
) -> Dict[str, Dict[str, int]]:
    """Generates the pairwise topic outputs for a given taxonomy.

    Args:
        taxonomy (pd.DataFrame): A dataframe with the taxonomy.
        entity_groupings (Dict[str, Dict[str, Sequence[str]]]): A dictionary with the entity groupings.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary with the pairwise topic outputs.
    """
    output = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for area, discipline_data in entity_groups.items():
        for discipline, topics in discipline_data.items():
            discipline_tax = taxonomy.loc[topics]
            topic_pairs = list(combinations(discipline_tax.index, 2))
            for level_col in [x for x in discipline_tax.columns if "Level" in x]:
                pairs = list(combinations(discipline_tax[level_col], 2))
                pairs_true = list(
                    compress(
                        topic_pairs, [True if x[0] == x[1] else False for x in pairs]
                    )
                )
                prop = np.sum([x[0] == x[1] for x in pairs]) / len(pairs)
                output[area][discipline][level_col] = {
                    "prop": prop,
                    "pairs": pairs_true,
                }
    return output


def generate_pairwise_subtopic_depths(
    taxonomy: pd.DataFrame,
    entity_groups: Dict[str, Dict[str, Dict[str, Sequence[str]]]],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Generates the pairwise subtopic outputs for a given taxonomy.

    Args:
        taxonomy (pd.DataFrame): A dataframe with the taxonomy.
        entity_groupings (Dict[str, Dict[str, Dict[str, Sequence[str]]]]): A dictionary
            with the entity groupings.

    Returns:
        Dict[str, Dict[str, Dict[str, int]]]: A dictionary with the pairwise subtopic
            outputs.
    """
    output = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    for area, discipline_data in entity_groups.items():
        for discipline, topics in discipline_data.items():
            for topic, subtopics in topics.items():
                topic_tax = taxonomy.loc[subtopics]
                subtopic_pairs = list(combinations(topic_tax.index, 2))
                for level_col in [x for x in topic_tax.columns if "Level" in x]:
                    pairs = list(combinations(topic_tax[level_col], 2))
                    pairs_true = list(
                        compress(
                            subtopic_pairs,
                            [True if x[0] == x[1] else False for x in pairs],
                        )
                    )
                    prop = np.sum([x[0] == x[1] for x in pairs]) / len(pairs)
                    output[area][discipline][topic][level_col] = {
                        "prop": prop,
                        "pairs": pairs_true,
                    }
    return output


ENTITIES = taxonomy["ENTITIES"]
SIMULATION_RATIOS = {
    "sample": ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"],
    "noise": ["00", "05", "07", "08", "11", "14", "18", "24", "31", "39", "51"],
}


# SIMULATION = "noise"
# OUTPUT_DIR = PROJECT_DIR / "outputs" / "figures" / "taxonomy_validation" / "simulations"
# (OUTPUT_DIR / SIMULATION).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    logger.info("Loading data...")
    entity_groups_topics = get_topic_groups()
    entity_groups_subtopics = get_subtopic_groups()
    embeddings = get_entity_embeddings(discarded=False)
    embeddings_discarded = get_entity_embeddings(discarded=True)
    embeddings = pd.concat([embeddings, embeddings_discarded])
    embeddings = {k: v for k, v in zip(embeddings.index, embeddings.to_numpy())}

    parser = argparse.ArgumentParser(
        description="Make histograms of a given taxonomy level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--taxonomy",
        nargs="+",
        help=(
            "The type of taxonomy to use. Can be a single taxonomy or a sequence \
            of taxonomies, and accepts 'community', 'centroids' or 'imbalanced' as tags."
        ),
        required=True,
    )

    parser.add_argument(
        "--simulation",
        nargs="+",
        help=(
            "The type of simulation to use. Can be a single simulation or a sequence \
            of simulations, and accepts 'sample' or/and 'noise' as tags."
        ),
        required=True,
    )

    parser.add_argument("--save", help="Whether to save the plot.", action="store_true")

    args = parser.parse_args()

    for simulation in args.simulation:
        logger.info("Loading simulation outputs...")
        simulations_dictionary = get_simulation_outputs(simulation=simulation)
        simul_str = "sample" if simulation == "sample" else "ratio"
        logger.info(f"Loaded outputs for {simulation} simulation.")

        for taxonomy in args.taxonomy:
            logger.info(f"Building datasets for the {taxonomy} taxonomy.")

            for ratio in SIMULATION_RATIOS[simulation]:
                logger.info(
                    f"{simulation} - {taxonomy} - Getting simulation data for ratio {ratio}."
                )
                key = [
                    x
                    for x in simulations_dictionary.keys()
                    if taxonomy in x and f"_{ratio}_" in x
                ]
                if len(key) == 1:
                    key = key[0]
                else:
                    raise ValueError(f"No single key found: {key}")

                logger.info(
                    f"{simulation} - {taxonomy} - Getting topic sizes for ratio {ratio}."
                )
                topic_size_df = get_topic_sizes(
                    simulations_dictionary[key], taxonomy, simulation, ratio
                )
                if args.save:
                    topic_size_df.to_parquet(
                        "s3://aria-mapping/outputs/simulations/{}/metrics/sizes/class_{}_{}_{}.parquet".format(
                            simulation, taxonomy, simul_str, ratio
                        )
                    )

                logger.info(
                    f"{simulation} - {taxonomy} - Getting centroid distances for ratio {ratio}."
                )
                centroid_distances_df = get_centroid_distances(
                    df=simulations_dictionary[key], embeddings=embeddings
                )
                if args.save:
                    centroid_distances_df.to_parquet(
                        "s3://aria-mapping/outputs/simulations/{}/metrics/distances/class_{}_{}_{}.parquet".format(
                            simulation, taxonomy, simul_str, ratio
                        )
                    )

                logger.info(
                    f"{simulation} - {taxonomy} - Getting topic pairwise depths for ratio {ratio}."
                )
                topic_pairwise_depths = generate_pairwise_topic_depths(
                    taxonomy=simulations_dictionary[key],
                    entity_groups=entity_groups_topics,
                )
                if args.save:
                    s3 = boto3.client("s3")
                    s3.put_object(
                        Body=json.dumps(topic_pairwise_depths),
                        Bucket=BUCKET_NAME,
                        Key="outputs/simulations/{}/metrics/pairwise/topic_class_{}_{}_{}.json".format(
                            simulation, taxonomy, simul_str, ratio
                        ),
                    )

                logger.info(
                    f"{simulation} - {taxonomy} - Getting subtopic pairwise depths for ratio {ratio}."
                )
                subtopic_pairwise_depths = generate_pairwise_subtopic_depths(
                    taxonomy=simulations_dictionary[key],
                    entity_groups=entity_groups_subtopics,
                )
                if args.save:
                    s3 = boto3.client("s3")
                    s3.put_object(
                        Body=json.dumps(subtopic_pairwise_depths),
                        Bucket=BUCKET_NAME,
                        Key="outputs/simulations/{}/metrics/pairwise/subtopic_class_{}_{}_{}.json".format(
                            simulation, taxonomy, simul_str, ratio
                        ),
                    )
