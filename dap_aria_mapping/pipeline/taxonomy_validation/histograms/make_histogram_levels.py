import pandas as pd
import argparse
import altair as alt
from nesta_ds_utils.viz.altair import formatting, saving

formatting.setup_theme()

from toolz import pipe
from dap_aria_mapping import logger, PROJECT_DIR
from typing import Union, Sequence
from functools import partial

from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
    get_semantic_taxonomy,
)
from dap_aria_mapping.getters.openalex import get_openalex_works, get_openalex_entities
from dap_aria_mapping.utils.entity_selection import get_sample, filter_entities
from dap_aria_mapping.utils.topic_names import *


def dataframe_to_histogram_df(
    dataframe: pd.DataFrame, level: int, n_top: int = 20
) -> pd.DataFrame:
    """Transforms a named taxonomy for use in altair histograms.

    Args:
        dataframe (pd.DataFrame): A named taxonomy.
        level (int): The level of the taxonomy to use.
        n_top (int, optional): The number of top topics to use. Defaults to 20.

    Returns:
        pd.DataFrame: A dataframe with the top n entities and their density.
    """
    dataframe = dataframe.loc[
        dataframe["Level_{}_Entity_Names".format(str(level))] != "E: "
    ]
    return (
        dataframe.value_counts(
            "Level_{}_Entity_Names".format(str(level)), normalize=True
        )
        .reset_index()
        .rename(columns={0: "density"})
        .iloc[:n_top]
    )


def name_topics(
    dataframe: pd.DataFrame,
    journal_entities: Dict[str, Sequence[str]],
    levels: Union[int, Sequence[int]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Create a named taxonomy from a list of most frequent entities.

    Args:
        dataframe (pd.DataFrame): A taxonomy dataframe, with at least one
            Level column and one Entity column.
        journal_entities (Dict[str, Sequence[str]]): A dictionary
            of journal to entities.
        levels (Union[int, Sequence[int]]): The number of levels
            to name.
        args (argparse.Namespace): The arguments passed to the script.

    Returns:
        pd.DataFrame: A taxonomy dataframe with named levels.
    """
    named_taxonomy = deepcopy(dataframe)
    level_range = (
        levels if isinstance(levels, list) else list(range(1, 1 + int(levels)))
    )
    for level in level_range:
        clust_counts = get_level_entity_counts(journal_entities, dataframe, level)
        cluster_names = get_cluster_entity_names(clust_counts, num_entities=args.n_top)

        named_taxonomy["Level_{}_Entity_Names".format(str(level))] = named_taxonomy[
            "Level_{}".format(str(level))
        ].map(cluster_names)

        named_taxonomy.replace(
            "Journals: $|Entities: $", "Other", regex=True, inplace=True
        )
    return named_taxonomy


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
        "--level",
        nargs="+",
        help="The level of the taxonomy to use. Can be a single level or a sequence of levels.",
        required=True,
    )

    parser.add_argument(
        "--n_top",
        type=int,
        help="The number of top topics to show. Defaults to 20.",
        default=20,
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
        taxonomies.append(["Network", cooccur_taxonomy])
    if "centroids" in args.taxonomy:
        semantic_centroids_taxonomy = get_semantic_taxonomy("centroids")
        taxonomies.append(["Semantic (Centroids)", semantic_centroids_taxonomy])
    if "imbalanced" in args.taxonomy:
        semantic_kmeans_taxonomy = get_semantic_taxonomy("imbalanced")
        taxonomies.append(["Semantic (Imbalanced)", semantic_kmeans_taxonomy])

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

    logger.info("Building named taxonomies")
    levels = [int(x) for x in args.level]
    named_taxonomies = [
        [name, name_topics(taxonomy, journal_entities, max(levels), args)]
        for name, taxonomy in taxonomies
    ]

    for level in levels:
        chart = alt.vconcat(
            *[
                (
                    alt.Chart(
                        dataframe_to_histogram_df(df, level, n_top=args.n_top),
                        title="Main Clusters in Level {} - {}".format(str(level), name),
                    )
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "Level_{}_Entity_Names".format(str(level)),
                            sort="-y",
                            title=None,
                            axis=alt.Axis(labelAngle=25, labelLimit=200),
                        ),
                        y=alt.Y("density", axis=alt.Axis(format=".2f")),
                    )
                    .properties(width=800, height=100)
                )
                for name, df in named_taxonomies
            ]
        )

        if args.save:
            chart.save(
                PROJECT_DIR
                / "outputs"
                / "figures"
                / "taxonomy_validation"
                / "taxonomy_level_{}.html".format(str(level))
            )
