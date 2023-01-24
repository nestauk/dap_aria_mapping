import pandas as pd
import argparse, pickle, boto3
from toolz import pipe
from dap_aria_mapping import logger, PROJECT_DIR, BUCKET_NAME
from typing import Union, Sequence
from functools import partial

from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
    get_semantic_taxonomy,
)
from dap_aria_mapping.getters.openalex import get_openalex_works, get_openalex_entities
from dap_aria_mapping.utils.entity_selection import get_sample, filter_entities
from dap_aria_mapping.utils.labelling import *


def save_labels(taxonomy: str, label_type: str, level: int, labels: Dict) -> None:
    """Save the labels for a given taxonomy level.

    Args:
        taxonomy (str): A string representing the taxonomy class.
            Can be 'cooccur', 'semantic' or 'semantic_kmeans'.
        label_type (str): A string representing the labels type.
        level (int): The level of the taxonomy to use.
        labels (Dict): A dictionary of the labels.
    """
    title = "outputs/topic_labels/labels_class_{}_labtype_{}_level_{}.pkl".format(
        taxonomy, label_type, str(level)
    )
    s3 = boto3.client("s3")
    s3.put_object(
        Body=pickle.dumps(labels),
        Bucket=BUCKET_NAME,
        Key=title,
    )


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
            of taxonomies, and accepts 'cooccur', 'centroids' or 'imbalanced' as tags."
        ),
        required=True,
    )

    parser.add_argument(
        "--label_type",
        nargs="+",
        help="The type of labels to use. Can be 'entity', 'journal', or both.",
        default="entity",
    )

    parser.add_argument(
        "--n_top",
        type=int,
        help="The number of top elements to show. Defaults to 3.",
        default=3,
    )

    parser.add_argument(
        "--levels", nargs="+", help="The levels of the taxonomy to use.", required=True
    )

    parser.add_argument(
        "--n_articles",
        type=int,
        help="The number of articles to use. If -1, all articles are used.",
    )

    parser.add_argument("--save", help="Whether to save the plot.", action="store_true")

    args = parser.parse_args()

    logger.info("Loading data - taxonomy")
    if not isinstance(args.taxonomy, list):
        args.taxonomy = [args.taxonomy]
    taxonomies = []
    if "cooccur" in args.taxonomy:
        cooccur_taxonomy = get_cooccurrence_taxonomy()
        taxonomies.append(["cooccur", cooccur_taxonomy])
    if "centroids" in args.taxonomy:
        semantic_centroids_taxonomy = get_semantic_taxonomy("centroids")
        taxonomies.append(["centroids", semantic_centroids_taxonomy])
    if "imbalanced" in args.taxonomy:
        semantic_kmeans_taxonomy = get_semantic_taxonomy("kmeans_strict_imb")
        taxonomies.append(["imbalanced", semantic_kmeans_taxonomy])

    logger.info("Loading data - works")
    oa_works = get_openalex_works()

    logger.info("Loading data - entities")
    oa_entities = pipe(
        get_openalex_entities(),
        partial(get_sample, score_threshold=80, num_articles=args.n_articles),
        partial(filter_entities, min_freq=10, max_freq=1_000_000, method="absolute"),
    )

    logger.info("Building dictionary - journal to entities")
    journal_entities = get_journal_entities(oa_works, oa_entities)

    for taxlabel, taxonomy in taxonomies:
        logger.info("Starting taxonomy {}".format(taxlabel))
        for level in [int(x) for x in args.levels]:
            logger.info("Starting level {}".format(str(level)))
            logger.info("Building dictionary - entity to journals")
            clust_counts = get_level_entity_counts(
                journal_entities, taxonomy, level=level
            )

            if "entity" in args.label_type:
                logger.info("Building dictionary - entity to labels")
                labels = get_cluster_entity_labels(
                    clust_counts, num_entities=args.n_top
                )
                if args.save:
                    logger.info("Saving dictionary as pickle")
                    save_labels(
                        taxonomy=taxlabel,
                        label_type="entity",
                        level=level,
                        labels=labels,
                    )

            if "journal" in args.label_type:
                logger.info("Building dictionary - journal to labels")
                cluster_journal_ranks = get_cluster_journal_ranks(
                    clust_counts, journal_entities, output="relative"
                )
                labels = get_cluster_journal_labels(
                    cluster_journal_ranks, num_labels=args.n_top
                )
                if args.save:
                    logger.info("Saving dictionary as pickle")
                    save_labels(
                        taxonomy=taxlabel,
                        label_type="journal",
                        level=level,
                        labels=labels,
                    )
            logger.info("Finished level {}".format(str(level)))
        logger.info("Finished taxonomy {}".format(taxlabel))
