import numpy as np
import pandas as pd
import argparse
import plotly.express as px
from plotly.graph_objects import Figure as PlotlyFigure
from typing import (
    Union,
    Dict,
    Sequence,
)
from functools import partial, reduce
from copy import deepcopy
from tqdm import tqdm
from toolz import pipe
from dap_aria_mapping import logger, PROJECT_DIR


from dap_aria_mapping.getters.taxonomies import get_cooccurrence_taxonomy
from dap_aria_mapping.getters.openalex import get_openalex_works, get_openalex_entities
from dap_aria_mapping.utils.entity_selection import get_sample, filter_entities
from dap_aria_mapping.utils.topic_names import *


def propagate_sunburst_cluster_values(
    df: pd.DataFrame,
    level: int,
    cluster_name: Union[str, bool],
    entity_dict: Dict[int, Sequence[str]],
    entity_length: int = 250,
    tail_path: Sequence[str] = ["Entity"],
) -> Dict:
    """Propagate sunburst values from a dataframe of the named taxonomy.

    Args:
        df (pd.DataFrame): A dataframe of the named taxonomy.
        level (int): Level of the taxonomy to propagate values for.
        cluster_name (Union[str, bool]): The cluster name to propagate values for. If
            `False`, all cluster Names will be selected. If `str`, the specified cluster
            name will be selected.
        entity_dict (Dict[int, Sequence[str]]): A dictionary of the top entities to propagate
            values for.
        entity_length (int, optional): The length of top entities to propagate values for.
        tail_path (Sequence[str], optional): The tail path to propagate values for.
            Defaults to ["Entity"].

    Returns:
        Dict: A dictionary of the sunburst values.
    """

    df = df.loc[df["Entity"].isin(entity_dict[entity_length])]
    if isinstance(cluster_name, bool):
        if cluster_name is False:
            pass
    elif isinstance(cluster_name, str):
        df = df.loc[df["Level_{}".format(str(level))] == cluster_name]

    if level == 1:
        path = ["Level_1_Entity_Names"] + tail_path
    elif level != 1:
        path = [
            "Level_1_Entity_Names",
            "Level_{}_Entity_Names".format(str(level)),
        ] + tail_path

    fig = px.sunburst(df, path=path, values="Count")
    return {
        "ids": [fig.data[0]["ids"]],
        "labels": [fig.data[0]["labels"]],
        "parents": [fig.data[0]["parents"]],
        "values": [fig.data[0]["values"]],
    }


def build_cluster_sunburst(
    df: pd.DataFrame,
    entity_dict: Dict[int, Sequence[str]],
    tail_path: Sequence[str] = ["Entity"],
    save: Union[bool, str] = False,
) -> PlotlyFigure:
    """Build a sunburst plot from a dataframe of the named taxonomy.

    Args:
        df (pd.DataFrame): A dataframe of the named taxonomy.
        entity_dict (Dict[int, Sequence[str]]): A dictionary of the top
            entities to propagate values for.
        tail_path (Sequence[str], optional): The tail path to propagate values for.
            Defaults to ["Entity"].
        save (bool, optional): Whether to save the plot. Defaults to False.

    Returns:
        PlotlyFigure: A plotly figure of the sunburst plot.
    """

    fig = px.sunburst(
        df.loc[df["Entity"].isin(list(entity_dict.values())[0])],
        path=["Level_1_Entity_Names", "Entity"],
        values="Count",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    updatemenus = [
        {
            "buttons": [
                {
                    "args": [
                        propagate_sunburst_cluster_values(
                            df=df,
                            level=level,
                            cluster_name=False,
                            entity_dict=entity_dict,
                            entity_length=length,
                            tail_path=tail,
                        )
                    ],
                    "label": "{} - Top {} Entities - Level {}".format(
                        tail_name, str(length), str(level)
                    ),
                    "method": "restyle",
                }
                for tail_name, tail in zip(
                    ["Entities", "Journal & Entities"],
                    [
                        [element for element in tail_path if element != "Journal"],
                        tail_path,
                    ],
                )
                for length in entity_dict.keys()
                for level in range(1, 4)
            ],
            "direction": "down",
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "x": 0.0,
            "y": 1.02,
            "xanchor": "left",
            "yanchor": "top",
            "type": "dropdown",
        }
    ]
    fig = fig.update_layout(
        updatemenus=updatemenus,
        width=1600,
        height=1200,
        autosize=False,
        margin=dict(l=10, r=0, b=0, t=100),
        title_text="Sunburst of Entity Clusters to relevant entities",
        title_x=0.5,
        title_y=0.95,
    )

    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Entity Count: %{value}<br>",
    )

    if isinstance(save, bool):
        if save:
            fig.write_html(
                PROJECT_DIR / "outputs" / "figures" / "sunburst_clusters.html",
                include_plotlyjs="cdn",
            )
            return None
    elif isinstance(save, str):
        fig.write_html(
            PROJECT_DIR / "outputs" / "figures" / save,
            include_plotlyjs="cdn",
        )
        return None

    return fig


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Build a sunburst plot of clusters and their associated entities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--n_entities",
        nargs="+",
        help="A list of different entity numbers to showcase. ",
        required=True,
    )

    parser.add_argument(
        "--levels", nargs="+", help="The levels of the taxonomy to use.", required=True
    )

    parser.add_argument(
        "--n_articles",
        type=int,
        help="The number of articles to use. If -1, all articles are used. Defaults to 20000.",
        default=20000,
    )

    parser.add_argument("--save", help="Whether to save the plot.", action="store_true")

    args = parser.parse_args()

    logger.info("Loading data - taxonomy")
    cooccur_taxonomy = get_cooccurrence_taxonomy()

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

    logger.info("Building named taxonomy")
    cooccur_taxonomy_named = build_named_taxonomy(
        cooccur_taxonomy, journal_entities, entity_counts, entity_journal_counts
    )

    logger.info("Creating plot dataframe - clusters")
    cluster_entity_journal_df = [
        pipe(
            get_cluster_entity_journal_counts(
                taxonomy=cooccur_taxonomy,
                journal_entities=journal_entities,
                level=level,
            ),
            partial(dictionary_to_df, level=level),
        )
        for level in range(1, int(args.levels[-1]) + 1)
    ]

    cluster_entity_journal_df = reduce(
        lambda left, right: pd.merge(
            left,
            right[
                ["Entity", "Journal"] + [col for col in right.columns if "Level" in col]
            ],
            on=["Entity", "Journal"],
            how="left",
        ),
        cluster_entity_journal_df,
    )

    cluster_entity_journal_df = pd.merge(
        cluster_entity_journal_df,
        cooccur_taxonomy_named[
            [col for col in cooccur_taxonomy_named.columns if "Entity_Names" in col]
        ].reset_index(),
        on="Entity",
        how="left",
    )

    MAIN_ENTITIES = {
        int(num): (
            cluster_entity_journal_df.groupby("Entity")[["Count"]]
            .sum()
            .reset_index()
            .sort_values(by="Count", ascending=False)[: int(num)]["Entity"]
            .to_list()
        )
        for num in args.n_entities
    }

    if args.save:
        save = "sunburst_journal.html"
    else:
        save = False

    logger.info("Building plot - clusters")
    fig = build_cluster_sunburst(
        cluster_entity_journal_df, entity_dict=MAIN_ENTITIES, save=save
    )
