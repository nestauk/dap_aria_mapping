import pandas as pd
import argparse
import altair as alt
from nesta_ds_utils.viz.altair import formatting, saving

formatting.setup_theme()
from toolz import pipe
from dap_aria_mapping import logger, PROJECT_DIR
from typing import Union, Sequence

from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
    get_semantic_taxonomy,
)
from dap_aria_mapping.getters.openalex import get_openalex_works, get_openalex_entities
from dap_aria_mapping.utils.semantics import get_sample, filter_entities
from dap_aria_mapping.utils.validation import *


def dataframe_to_histogram_df(
    dataframe: pd.DataFrame, level: int, n_top: int = 20
) -> pd.DataFrame:
    """Transforms a labelled taxonomy for use in altair histograms.

    Args:
        dataframe (pd.DataFrame): A labelled taxonomy.
        level (int): The level of the taxonomy to use.
        n_top (int, optional): The number of top topics to use. Defaults to 20.

    Returns:
        pd.DataFrame: A dataframe with the top n entities and their density.
    """
    dataframe = dataframe.loc[
        dataframe["Level_{}_Entity_Labels".format(str(level))] != "E: "
    ]
    return (
        dataframe.value_counts(
            "Level_{}_Entity_Labels".format(str(level)), normalize=True
        )
        .reset_index()
        .rename(columns={0: "density"})
        .iloc[:n_top]
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Make histograms of a given taxonomy level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    cooccur_taxonomy = get_cooccurrence_taxonomy()
    semantic_taxonomy = get_semantic_taxonomy()
    semantic_kmeans_taxonomy = pipe(
        "kmeans_strict_imb",
        get_semantic_taxonomy,
        lambda df: df.rename(columns={"tag": "Entity"}) if "tag" in df.columns else df,
        lambda df: df.set_index("Entity"),
        lambda df: df.rename(
            columns={k: "Level_{}".format(str(v + 1)) for v, k in enumerate(df.columns)}
        ),
    )

    if "tag" in semantic_taxonomy.columns:
        semantic_taxonomy.rename(columns={"tag": "Entity"}, inplace=True)
    semantic_taxonomy.set_index("Entity", inplace=True)
    semantic_taxonomy.rename(
        columns={
            k: "Level_{}".format(str(v + 1))
            for v, k in enumerate(semantic_taxonomy.columns)
        },
        inplace=True,
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

    logger.info("Building dictionary - entity to counts")
    entity_counts = get_entity_counts(journal_entities)

    logger.info("Building dictionary - entity to journals")
    entity_journal_counts = get_entity_journal_counts(
        journal_entities, output="absolute"
    )

    logger.info("Building labelled taxonomies")
    cooccur_taxonomy_labelled = build_labelled_taxonomy(
        cooccur_taxonomy, journal_entities, entity_counts, entity_journal_counts
    )

    semantic_taxonomy_labelled = build_labelled_taxonomy(
        semantic_taxonomy, journal_entities, entity_counts, entity_journal_counts
    )

    semantic_kmeans_taxonomy_labelled = build_labelled_taxonomy(
        semantic_kmeans_taxonomy, journal_entities, entity_counts, entity_journal_counts
    )

    levels = args.level if isinstance(args.level, list) else list(args.level)

    for level in levels:
        chart = alt.vconcat(
            *[
                (
                    alt.Chart(
                        dataframe_to_histogram_df(df, level, n_top=args.n_top),
                        title="Main Clusters in Level {} - {}".format(
                            str(level), label
                        ),
                    )
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "Level_{}_Entity_Labels".format(str(level)),
                            sort="-y",
                            title=None,
                            axis=alt.Axis(labelAngle=25, labelLimit=200),
                        ),
                        y=alt.Y("density", axis=alt.Axis(format=".2f")),
                    )
                    .properties(width=800, height=100)
                )
                for label, df in zip(
                    ["Network", "Semantic (Centroids)", "Semantic (Imbalanced)"],
                    [
                        cooccur_taxonomy_labelled,
                        semantic_taxonomy_labelled,
                        semantic_kmeans_taxonomy_labelled,
                    ],
                )
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
