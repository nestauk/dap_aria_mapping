import pandas as pd
import argparse, json, boto3
from toolz import pipe
from dap_aria_mapping import logger, PROJECT_DIR, BUCKET_NAME
from functools import partial
import re

from dap_aria_mapping.getters.openalex import get_openalex_works, get_openalex_entities
from dap_aria_mapping.utils.entity_selection import get_sample, filter_entities
from dap_aria_mapping.utils.topic_names import *
from dap_aria_mapping.getters.simulations import get_simulation_outputs


def save_names(
    taxonomy: str, level: int, names: Dict, simulation: str, ratio: str
) -> None:
    """Save the topic names for a given taxonomy level.

    Args:
        taxonomy (str): A string representing the taxonomy class.
            Can be 'cooccur', 'semantic' or 'semantic_kmeans'.
        level (int): The level of the taxonomy to use.
        names (Dict): A dictionary of the topic names.
    """
    names = {str(k): v for k, v in names.items()}
    title = "outputs/simulations/{}/names/class_{}_{}_level_{}.json".format(
        simulation, taxonomy, ratio, str(level)
    )
    s3 = boto3.client("s3")
    s3.put_object(
        Body=json.dumps(names),
        Bucket=BUCKET_NAME,
        Key=title,
    )


def parse_simulations(
    simulations_dictionary: Dict[str, pd.DataFrame], taxonomy_class: str
) -> Dict[str, pd.DataFrame]:
    """Parse the simulation outputs into a dictionary of dataframes.

    Args:
        simulations_dictionary (Dict[str, pd.DataFrame]): A dictionary of simulation outputs.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary of dataframes.
    """
    subset_dictionary = {
        k: v for k, v in simulations_dictionary.items() if taxonomy_class in k
    }
    parsed_simulations = {}
    for simulation, df in subset_dictionary.items():
        ratio = re.search(r"(?:ratio_|sample_)(\d{2,3})_", simulation).group(0)
        parsed_simulations[ratio[:-1]] = df
    return parsed_simulations


if __name__ == "__main__":

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

    parser.add_argument(
        "--n_top",
        type=int,
        help="The number of top elements to show. Defaults to 3.",
        default=3,
    )

    parser.add_argument(
        "--n_articles",
        type=int,
        help="The number of articles to use. If -1, all articles are used.",
    )

    parser.add_argument("--save", help="Whether to save the plot.", action="store_true")

    args = parser.parse_args()

    logger.info("Loading data - works")
    oa_works = get_openalex_works()

    logger.info("Loading data - entities")
    oa_entities = pipe(
        get_openalex_entities(),
        partial(get_sample, score_threshold=70, num_articles=args.n_articles),
        partial(filter_entities, min_freq=0, max_freq=1_000_000, method="absolute"),
    )

    logger.info("Building dictionary - journal to entities")
    journal_entities = get_journal_entities(oa_works, oa_entities)

    del oa_works, oa_entities

    for simulation in args.simulation:
        logger.info("Starting simulation {}".format(simulation))
        simulations_dictionary = get_simulation_outputs(simulation=simulation)

        logger.info("Loading data - taxonomy")
        taxonomies = []
        for taxonomy in args.taxonomy:
            taxonomies.append(
                [taxonomy, parse_simulations(simulations_dictionary, taxonomy)]
            )

        for taxlabel, taxonomy_dict in taxonomies:
            logger.info("Starting taxonomy {}".format(taxlabel))
            for level in range(1, 6):
                logger.info("Starting level {}".format(str(level)))
                logger.info("Building dictionary - entity to journals")
                for ratio, taxonomy in taxonomy_dict.items():
                    clust_counts = get_level_entity_counts(
                        journal_entities, taxonomy, level=level
                    )

                    logger.info("Building dictionary - entity to names")
                    names = get_cluster_entity_names(
                        clust_counts, num_entities=args.n_top
                    )
                    if args.save:
                        logger.info("Saving dictionary as pickle")
                        save_names(
                            taxonomy=taxlabel,
                            level=level,
                            names=names,
                            simulation=simulation,
                            ratio=ratio,
                        )

                    logger.info("Finished level {}".format(str(level)))
                logger.info("Finished taxonomy {}".format(taxlabel))
