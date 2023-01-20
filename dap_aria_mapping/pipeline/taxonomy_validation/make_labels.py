import pandas as pd
import argparse, pickle, boto3
from toolz import pipe
from dap_aria_mapping import logger, PROJECT_DIR, BUCKET_NAME
from typing import Union, Sequence

from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
    get_semantic_taxonomy,
)
from dap_aria_mapping.getters.openalex import get_openalex_works, get_openalex_entities
from dap_aria_mapping.utils.semantics import get_sample, filter_entities
from dap_aria_mapping.utils.validation import *


def save_labels(taxonomy_class: str, label_type: str, level: int, labels: Dict) -> None:
    """Save the labels for a given taxonomy level.

    Args:
        taxonomy_class (str): A string representing the taxonomy class.
            Can be 'cooccur', 'semantic' or 'semantic_kmeans'.
        label_type (str): A string representing the labels type.
        level (int): The level of the taxonomy to use.
        labels (Dict): A dictionary of the labels.
    """
    title = "outputs/topic_labels/labels_class_{}_labtype_{}_level_{}.pkl".format(
        taxonomy_class, label_type, str(level)
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
        "--taxonomy_class",
        help="The type of taxonomy to use. Can be 'cooccur', 'semantic' or 'semantic_kmeans'.",
        default="cooccur",
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
    if args.taxonomy_class == "cooccur":
        taxonomy = get_cooccurrence_taxonomy()
    elif args.taxonomy_class == "semantic":
        taxonomy = pipe(
            "centroids",
            get_semantic_taxonomy,
            lambda df: df.rename(columns={"tag": "Entity"})
            if "tag" in df.columns
            else df,
            lambda df: df.set_index("Entity"),
            lambda df: df.rename(
                columns={
                    k: "Level_{}".format(str(v + 1)) for v, k in enumerate(df.columns)
                }
            ),
        )
    elif args.taxonomy_class == "semantic_kmeans":
        taxonomy = pipe(
            "kmeans_strict_imb",
            get_semantic_taxonomy,
            lambda df: df.rename(columns={"tag": "Entity"})
            if "tag" in df.columns
            else df,
            lambda df: df.set_index("Entity"),
            lambda df: df.rename(
                columns={
                    k: "Level_{}".format(str(v + 1)) for v, k in enumerate(df.columns)
                }
            ),
        )

    logger.info("Loading data - works")
    oa_works = get_openalex_works()

    logger.info("Loading data - entities")
    oa_entities = get_openalex_entities()
    oa_entities = get_sample(
        entities=oa_entities, score_threshold=80, num_articles=args.n_articles
    )

    logger.info("Building dictionary - journal to entities")
    journal_entities = get_journal_entities(oa_works, oa_entities)

    for level in [int(x) for x in args.levels]:
        logger.info("Starting level {}".format(str(level)))
        logger.info("Building dictionary - entity to journals")
        clust_counts = get_level_entity_counts(journal_entities, taxonomy, level=level)

        if "entity" in args.label_type:
            logger.info("Building dictionary - entity to labels")
            labels = get_cluster_entity_labels(clust_counts, num_entities=args.n_top)
            if args.save:
                logger.info("Saving dictionary as pickle")
                save_labels(
                    taxonomy_class=args.taxonomy_class,
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
                    taxonomy_class=args.taxonomy_class,
                    label_type="journal",
                    level=level,
                    labels=labels,
                )
        logger.info("Finished level {}".format(str(level)))
