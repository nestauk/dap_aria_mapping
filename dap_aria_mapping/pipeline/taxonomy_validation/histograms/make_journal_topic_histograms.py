# %%
import pandas as pd
import argparse, pickle, json
from nesta_ds_utils.viz.altair import formatting, saving

formatting.setup_theme()
from toolz import pipe
from dap_aria_mapping import logger, PROJECT_DIR
from functools import partial, reduce

from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
    get_semantic_taxonomy,
)

from dap_aria_mapping.getters.openalex import get_openalex_works, get_openalex_entities
from dap_aria_mapping.utils.entity_selection import get_sample, filter_entities
from dap_aria_mapping.utils.topic_names import *
from dap_aria_mapping.pipeline.taxonomy_validation.histograms.journal_topic_utils import *


def dictionary_to_df(dictionary: Dict, level: int) -> pd.DataFrame:
    """Convert a dictionary of the named taxonomy to a dataframe.

    Args:
        dictionary (Dict): A dictionary of the named taxonomy.
        level (int): The level of the taxonomy to convert.

    Returns:
        pd.DataFrame: A new dataframe of the named taxonomy.
    """
    outdf = []
    for clust, entities in dictionary.items():
        for entity, journals in entities.items():
            for journal, count in journals.items():
                outdf.append((clust, entity, journal, count))
    return pd.DataFrame(
        outdf, columns=["Level_{}".format(str(level)), "Entity", "Journal", "Count"]
    )

s
def output_charts(
    df: pd.DataFrame,
    taxonomy_class: str,
    level: int,
    parent_topic: str,
    sample: Union[list, int],
) -> None:
    """Output charts for a given taxonomy class, level, and parent topic.

    Args:
        df (pd.DataFrame): A dataframe of Entity x Journal x Level assignments.
        taxonomy_class (str): The taxonomy class to output charts for.
        level (int): The level of the taxonomy to output charts for.
        parent_topic (str): The parent topic to output charts for.
        sample (Union[list, int]): The sample of entities to output charts for.
    """

    if "unary" in args.histogram:
        logger.info(f"Building unary data - {taxonomy_class} - level {level}")
        journal_unary_df = build_unary_data(
            df=df,
            taxonomy_class=taxonomy_class,
            level=level,
            parent_topic=parent_topic,
            threshold=args.threshold,
            sample=sample,
        )

        logger.info(f"Creating histogram - {taxonomy_class} - level {level}")
        make_unary_histogram(
            journal_unary_df,
            taxonomy_class=taxonomy_class,
            level=level,
            parent_topic=parent_topic,
            save=args.save,
        )

        logger.info(f"Creating heatmap - {taxonomy_class} - level {level}")
        make_unary_binary_chart(
            journal_unary_df,
            taxonomy_class=taxonomy_class,
            level=level,
            parent_topic=parent_topic,
            save=args.save,
        )

    if "frequency" in args.histogram:
        logger.info(f"Building frequency data - {taxonomy_class} - level {level}")
        journal_freq_df = build_frequency_data(
            df=df,
            taxonomy_class=taxonomy_class,
            level=level,
            parent_topic=parent_topic,
            threshold=1.0,
            sample=sample,
        )

        logger.info(f"Creating frequency histogram - {taxonomy_class} - level {level}")
        make_freq_barplot(
            journal_freq_df,
            taxonomy_class=taxonomy_class,
            level=level,
            parent_topic=parent_topic,
            save=args.save,
        )

    if "tfidf" in args.histogram:
        logger.info(f"Building TFIDF data - {taxonomy_class} - level {level}")
        journal_freq_df = build_tfidf_data(
            df=df,
            taxonomy_class=taxonomy_class,
            level=level,
            parent_topic=parent_topic,
            threshold=1.0,
            sample=sample,
        )

        logger.info(f"Creating TFIDF histogram - {taxonomy_class} - level {level}")
        make_tfidf_barplot(
            journal_freq_df,
            taxonomy_class=taxonomy_class,
            level=level,
            parent_topic=parent_topic,
            save=args.save,
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
        "--histogram",
        nargs="+",
        help=(
            "The type of histogram to use. Can be a single histogram or a sequence \
            of histograms, and accepts 'unary', 'frequency' or 'tfidf' as tags."
        ),
        required=True,
    )

    parser.add_argument(
        "--level",
        nargs="+",
        help=(
            "The level of the taxonomy to use. If the level is more than 1, flag `parent_topic` is required."
            "Can be a single level or a sequence of levels, which requires a sequence of parent_topics."
        ),
        required=True,
    )

    parser.add_argument(
        "--parent_topic",
        nargs="+",
        help="The parent topic of the taxonomy to use. If the level is more than 1, this is required. Can be"
        "a single parent topic or a sequence of parent topics, which requires a sequence of levels. Alternatively,"
        "if `all` is passed with this flag, histograms are created for all possible parent topics.",
    )

    parser.add_argument(
        "--sample",
        help="If `main`, it uses the 50 most frequent journals. Otherwise it accepts an integer for a "
        "random sample of journals. It also accepts `main{int}` to use the {int} number of most frequent"
        "journals within a given level / parent_topic combination.",
        default="main",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        help="The threshold of relevant topics, as a percentage of total journal-entity-topic assignments."
        "Only topics above this threshold are plotted, and only works for unary plots.",
        default=0.8,
    )

    parser.add_argument(
        "--n_articles",
        type=int,
        help="The number of articles to use. If -1, all articles are used.",
    )

    parser.add_argument("--save", help="Whether to save the plot.", action="store_true")

    args = parser.parse_args()

    logger.info("Loading data - taxonomy")
    taxonomies = []
    if "cooccur" in args.taxonomy:
        cooccur_taxonomy = get_cooccurrence_taxonomy()
        taxonomies.append(["cooccur", cooccur_taxonomy])
    if "centroids" in args.taxonomy:
        semantic_centroids_taxonomy = get_semantic_taxonomy("centroids")
        taxonomies.append(["centroids", semantic_centroids_taxonomy])
    if "imbalanced" in args.taxonomy:
        semantic_kmeans_taxonomy = get_semantic_taxonomy("imbalanced")
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

    logger.info("Building dictionary - entity to counts")
    entity_counts = get_entity_counts(journal_entities)

    logger.info("Building dictionary - entity to journals")
    entity_journal_counts = get_entity_journal_counts(
        journal_entities, output="absolute"
    )

    levels = (
        [int(x) for x in args.level]
        if isinstance(args.level, list)
        else list(int(args.level))
    )
    parent_topics = (
        args.parent_topic
        if isinstance(args.parent_topic, list)
        else list(args.parent_topic)
    )

    logger.info("Building named taxonomies")
    named_taxonomies = [
        [
            name,
            build_named_taxonomy(
                taxonomy,
                journal_entities,
                entity_counts,
                entity_journal_counts,
                max(levels),
            ),
        ]
        for name, taxonomy in taxonomies
    ]

    for taxonomy_class, taxonomy_df in named_taxonomies:
        logger.info("Building dictionary - cluster to entity x journal")
        cluster_entity_journal_df = [
            pipe(
                get_cluster_entity_journal_counts(
                    taxonomy=taxonomy_df,
                    journal_entities=journal_entities,
                    level=level,
                ),
                partial(dictionary_to_df, level=level),
            )
            for level in range(1, 1 + max(levels))
        ]
        cluster_entity_journal_df = reduce(
            lambda left, right: pd.merge(
                left,
                right[
                    ["Entity", "Journal"]
                    + [col for col in right.columns if "Level" in col]
                ],
                on=["Entity", "Journal"],
                how="left",
            ),
            cluster_entity_journal_df,
        )

        for level, parent_topic in list(zip(levels, parent_topics)):
            if parent_topic == "all":
                for partopic in cluster_entity_journal_df[
                    f"Level_{str(level-1)}"
                ].unique():
                    output_charts(
                        df=cluster_entity_journal_df,
                        taxonomy_class=taxonomy_class,
                        level=level,
                        parent_topic=partopic,
                        sample=args.sample,
                    )
            else:
                output_charts(
                    df=cluster_entity_journal_df,
                    taxonomy_class=taxonomy_class,
                    level=level,
                    parent_topic=parent_topic,
                    sample=args.sample,
                )
