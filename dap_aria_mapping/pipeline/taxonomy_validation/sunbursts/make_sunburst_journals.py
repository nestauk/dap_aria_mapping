# %%
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

# %%
from dap_aria_mapping.getters.taxonomies import get_cooccurrence_taxonomy
from dap_aria_mapping.getters.openalex import get_openalex_works, get_openalex_entities
from dap_aria_mapping.utils.entity_selection import get_sample, filter_entities
from dap_aria_mapping.utils.topic_names import *


def propagate_sunburst_journal_values(
    df: pd.DataFrame,
    journal: Union[str, bool],
    level: Union[Union[int, str], Sequence[Union[int, str]]] = 1,
) -> Dict:
    """Propagate sunburst values from a dataframe of the named taxonomy.
        The sunburst values are then used to create a sunburst plot, and the
        resulting data parameters are used to build plotting alternatives
        using plotly's `updatemenus`.

    Args:
        df (pd.DataFrame): A dataframe of the named taxonomy.
        journal (Union[str, bool]): The journal to propagate values for. If `True`,
            a random journal will be selected. If `False`, all journals will be
            selected. If `str`, the specified journal will be selected.
        level (Union[Union[int, str], Sequence[Union[int, str]]], optional):
            The level(s) to propagate values for. Defaults to 1.

    Returns:
        Dict: A dictionary of the sunburst values. Note these are not the
            values used to create the sunburst plot, but the values used
            by plotly to build the plot.
    """
    if isinstance(level, int):
        columns = ["Level_{}_Entity_Names".format(str(level))]
    elif isinstance(level, list):
        columns = ["Level_{}_Entity_Names".format(str(x)) for x in level]

    if isinstance(journal, bool):
        if journal:
            journal = np.random.choice(df["Main_Journal"].unique(), 1)[0]
            df = df.loc[df["Main_Journal"] == journal]
        else:
            df = df.loc[lambda df: df["Main_Journal"].isin(MAIN_JOURNALS)]
    elif isinstance(journal, str):
        df = df.loc[df["Main_Journal"] == journal]

    path = ["Main_Journal"] + columns + ["Entity"]

    fig = px.sunburst(
        df,
        path=path,
        values="Entity_Count",
    )
    return {
        "ids": [fig.data[0]["ids"]],
        "names": [fig.data[0]["names"]],
        "parents": [fig.data[0]["parents"]],
        "values": [fig.data[0]["values"]],
    }


def build_journal_sunburst(
    df: pd.DataFrame,
    level: Union[Union[int, str], Sequence[Union[int, str]]] = 1,
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
    if isinstance(level, int):
        columns = ["Level_{}_Entity_Names".format(str(level))]
    elif isinstance(level, list):
        columns = ["Level_{}_Entity_Names".format(str(x)) for x in level]

    path = ["Main_Journal"] + columns + ["Entity"]

    df = df.loc[lambda df: df["Main_Journal"].isin(MAIN_JOURNALS)]

    fig = px.sunburst(
        df,
        path=path,
        values="Entity_Count",
        hover_name="Entity",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        maxdepth=1,
    )

    updatemenus = [
        {
            "active": 0,
            "buttons": [
                {
                    "method": "update",
                    "name": "Top Publishing Journals",
                    "args": [
                        propagate_sunburst_journal_values(
                            df, journal=False, level=level
                        )
                    ],
                }
            ]
            + [
                {
                    "method": "update",
                    "name": journal,
                    "args": [
                        propagate_sunburst_journal_values(
                            df, journal=journal, level=level
                        )
                    ],
                }
                for journal in tqdm(sorted(df["Main_Journal"].unique()))
            ],
            "showactive": True,
            "x": 0.02,
            "y": 1.05,
            "pad": {"r": 10, "t": 10},
            "xanchor": "left",
            "yanchor": "top",
        },
        {
            "buttons": [
                {"args": [{"maxdepth": level}], "name": name, "method": "restyle"}
                for name, level in zip(
                    ["No Level"] + ["Level {}".format(str(x)) for x in level],
                    [1] + [int(x) + 1 for x in level],
                )
            ]
            + [
                {
                    "args": [{"maxdepth": int(level[-1]) + 2}],
                    "name": "Entities",
                    "method": "restyle",
                }
            ],
            "direction": "right",
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "x": 0.42,
            "y": 1.05,
            "xanchor": "left",
            "yanchor": "top",
            "type": "buttons",
        },
    ]

    fig = fig.update_layout(
        updatemenus=updatemenus,
        width=1600,
        height=1200,
        autosize=False,
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig.update_traces(
        hovertemplate="<b>%{name}</b><br>Entity Count: %{value}<br>",
    )

    if isinstance(save, bool):
        if save:
            fig.write_html(
                PROJECT_DIR / "outputs" / "figures" / "sunburst.html",
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
        description="Build a sunburst plot of the taxonomy with journals as focal points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--levels", nargs="+", help="The levels of the taxonomy to use.", required=True
    )

    parser.add_argument(
        "--n_journals",
        type=int,
        help="The number of journals to use. Defaults to 50.",
        default=50,
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
        df=cooccur_taxonomy,
        journal_entities=journal_entities,
        entity_counts=entity_counts,
        entity_journal_counts=entity_journal_counts,
        levels=int(args.levels[-1]),
    )

    logger.info("Creating list of relevant journals")
    MAIN_JOURNALS = [
        x
        for x in (
            cooccur_taxonomy_named["Main_Journal"]
            .value_counts()[: args.n_journals]
            .index.to_list()
        )
        if "Other" not in x
    ]

    if args.save:
        save = "sunburst_journal.html"
    else:
        save = False

    fig = build_journal_sunburst(
        cooccur_taxonomy_named.reset_index(), level=args.levels, save=save
    )
# %%
