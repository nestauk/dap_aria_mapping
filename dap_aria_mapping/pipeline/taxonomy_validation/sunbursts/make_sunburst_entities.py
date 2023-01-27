
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
from copy import deepcopy
from tqdm import tqdm
from toolz import pipe
from functools import partial
from dap_aria_mapping import logger, PROJECT_DIR


from dap_aria_mapping.getters.taxonomies import get_cooccurrence_taxonomy
from dap_aria_mapping.getters.openalex import get_openalex_works, get_openalex_entities
from dap_aria_mapping.utils.entity_selection import get_sample, filter_entities
from dap_aria_mapping.utils.topic_names import *


def propagate_sunburst_entity_values(
    df: pd.DataFrame, entity_dict: Union[str, bool], entity_length: int = 25
) -> Dict:
    """Propagate sunburst values from a dataframe of the named taxonomy.
        The sunburst values are then used to create a sunburst plot, and the
        resulting data parameters are used to build plotting alternatives
        using plotly's `updatemenus`.
    Args:
        df (pd.DataFrame): A dataframe of the named taxonomy.
        entity_dict (Union[str, bool]): The entity to propagate values for. If `True`,
            a random entity will be selected. If `False`, all entities will be
            selected. If `str`, the specified entity will be selected.
        entity_length (int, optional): The length of top entities to propagate values for.
            Defaults to 25.

    Returns:
        Dict: A dictionary of the sunburst values.
    """
    df = df.loc[df["Entity"].isin(entity_dict[entity_length])]
    fig = px.sunburst(
        df,
        path=["Entity", "Journal"],
        values="Entity_Count",
    )
    return {
        "ids": [fig.data[0]["ids"]],
        "names": [fig.data[0]["names"]],
        "parents": [fig.data[0]["parents"]],
        "values": [fig.data[0]["values"]],
    }


def build_entity_sunburst(
    df: pd.DataFrame,
    entity_dict: Dict[str, Sequence[str]],
    save: Union[bool, str] = False,
) -> PlotlyFigure:
    """Build a sunburst plot from a dataframe of the named taxonomy.

    Args:
        df (pd.DataFrame): A dataframe of the named taxonomy.
        level (Union[Union[int, str], Sequence[Union[int, str]]], optional):
            The level(s) to propagate values for. Defaults to 1.
        save (bool, optional): Whether to save the plot. Defaults to False.

    Returns:
        PlotlyFigure: A plotly figure of the sunburst plot.
    """
    fig = px.sunburst(
        df.loc[df["Entity"].isin(list(entity_dict.values())[0])],
        path=["Entity", "Journal"],
        values="Entity_Count",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    updatemenus = [
        {
            "buttons": [
                {
                    "method": "update",
                    "name": "Top {} Entities".format(str(length)),
                    "args": [
                        propagate_sunburst_entity_values(
                            df, entity_dict=entity_dict, entity_length=length
                        )
                    ],
                }
                for length in entity_dict.keys()
            ],
            "showactive": True,
        }
    ]

    fig = fig.update_layout(
        updatemenus=updatemenus,
        width=1600,
        height=1200,
        autosize=False,
        margin=dict(l=10, r=0, b=0, t=10),
    )

    fig.update_traces(hovertemplate="<b>%{name}</b><br>Entity Count: %{value}")

    if isinstance(save, bool):
        if save:
            fig.write_html(
                PROJECT_DIR / "outputs" / "figures" / "sunburst_entities.html",
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Build a sunburst plot of entities and their associated journals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--n_entities",
        nargs="+",
        help="The number of entities to showcase.",
        required=True,
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

    logger.info("Creating plot dataframe - real counts, not mainjournal ones")
    entity_counts_df = []
    for entity in cooccur_taxonomy_named.index:
        journal_counts = cooccur_taxonomy_named.loc[entity]["Journal_Counts"]
        if len(journal_counts) > 0:
            rows = []
            for journal, count in journal_counts.items():
                rows.append((entity, journal, count))
            entity_counts_df.append(rows)
    entity_counts_df = pipe(
        entity_counts_df,
        lambda dta: chain(*dta),
        list,
        lambda dta: pd.DataFrame(dta, columns=["Entity", "Journal", "Entity_Count"]),
    )

    MAIN_ENTITIES = {
        int(k): (
            entity_counts_df.groupby("Entity")[["Entity_Count"]]
            .sum()
            .reset_index()
            .sort_values(by="Entity_Count", ascending=False)[: int(k)]["Entity"]
            .to_list()
        )
        for k in args.n_entities
    }

    if args.save:
        save = "sunburst_entities.html"
    else:
        save = False

    fig = build_entity_sunburst(entity_counts_df, save=save, entity_dict=MAIN_ENTITIES)

