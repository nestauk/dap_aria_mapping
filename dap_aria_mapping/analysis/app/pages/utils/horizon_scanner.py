from dap_aria_mapping.getters.app_tables.horizon_scanner import (
    volume_per_year,
    novelty_per_year,
    get_novelty_documents,
    get_entity_document_lists,
    get_document_names,
    get_entities,
)
import streamlit as st
import pandas as pd
import polars as pl
from typing import List, Tuple
from itertools import chain
from collections import defaultdict


@st.cache_data(show_spinner="Loading data")
def load_overview_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str]]:
    """loads in the volume per year chart and does initial formatting that is not impacted by filters
    caches results so the data is not loaded each time a filter is run

    Returns:
        pl.DataFrame: total patents/publications per domain/area/topic per year, with names
        pl.DataFrame: same as above, but patent/publication counts are melted to long form
        List: unique domain names in dataset
    """
    volume_data = volume_per_year()

    # generate a list of the unique domain names to use as the filter
    unique_domains = volume_data["domain_name"].unique().to_list()
    unique_domains.insert(0, "All")
    # reformat the patent/publication counts to long form for the alignment chart
    alignment_data = volume_data.melt(
        id_vars=[
            "year",
            "topic",
            "topic_name",
            "area",
            "area_name",
            "domain",
            "domain_name",
        ],
        value_vars=["publication_count", "patent_count"],
    )
    alignment_data.columns = [
        "year",
        "topic",
        "topic_name",
        "area",
        "area_name",
        "domain",
        "domain_name",
        "doc_type",
        "count",
    ]

    return (
        volume_data,
        alignment_data,
        unique_domains,
    )


@st.cache_data(show_spinner="Loading novelty data")
def load_novelty_data():
    novelty_data = novelty_per_year()
    novelty_docs = get_novelty_documents()
    document_names = get_document_names()
    entity_dict = get_entity_document_lists()

    return novelty_data, novelty_docs, document_names, entity_dict


# Filter by domain
@st.cache_data(show_spinner="Filtering by domain")
def filter_volume_by_domain(
    domain: str,
    _volume_data: pl.DataFrame,
    _alignment_data: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
    """filters volume data, alignment data, and filter options based on a Domain selection

    Args:
        domain (str): domain selected by the filter
        _volume_data (pl.DataFrame): volume data for emergence chart
        _alignment_data (pl.DataFrame): alignment data for alignment chart
        _novelty_data (pl.DataFrame): novelty data for novelty chart

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str]]: updated dataframes filtered by a domain, and a list of unique areas to populate area filter
    """
    volume_data = _volume_data.filter(pl.col("domain_name") == domain)
    unique_areas = volume_data["area_name"].unique().to_list()
    alignment_data = _alignment_data.filter(pl.col("domain_name") == domain)
    return volume_data, alignment_data, unique_areas


@st.cache_data(show_spinner="Filtering by area")
def filter_volume_by_area(
    area: str,
    _volume_data: pl.DataFrame,
    _alignment_data: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
    """filters volume data, alignment data, and filter options based on an area selection

    Args:
        area (str): domain selected by the filter
        _volume_data (pl.DataFrame): volume data for emergence chart
        _alignment_data (pl.DataFrame): alignment data for alignment chart

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str]]: updated dataframes filtered by an area, and a list of unique topics to populate topic filter
    """
    volume_data = _volume_data.filter(pl.col("area_name") == area)
    alignment_data = _alignment_data.filter(pl.col("area_name") == area)
    unique_topics = volume_data["topic_name"].unique().to_list()
    return volume_data, alignment_data, unique_topics


@st.cache_data(show_spinner="Filtering by domain")
def filter_novelty_by_domain(
    domain: str,
    _novelty_data: pl.DataFrame,
) -> pl.DataFrame:
    novelty_data = _novelty_data.filter(pl.col("domain_name") == domain)
    return novelty_data


@st.cache_data(show_spinner="Filtering by area")
def filter_novelty_by_area(
    area: str,
    _novelty_data: pl.DataFrame,
) -> pl.DataFrame:
    novelty_data = _novelty_data.filter(pl.col("area_name") == area)
    return novelty_data


@st.cache_data
def group_emergence_by_level(
    _volume_data: pl.DataFrame, level: str, y_col: str
) -> pl.DataFrame:
    """groups the data for the emergence chart by the level specified by the filters

    Args:
        _volume_data (pl.DataFrame): data for backend of emergence chart
        level (str): level to view, specified by domain/area filters
        y_col (str): patents, publications, or all documents (specified by filter)

    Returns:
        pl.DataFrame: grouped emergence data for chart
    """
    q = (
        _volume_data.lazy()
        .with_columns(pl.col(level).cast(str))
        .groupby([level, "{}_name".format(level), "year"])
        .agg([pl.sum(y_col)])
        .filter(pl.any(pl.col("year").is_not_null()))
    )
    return q.collect()


@st.cache_data(show_spinner="Filtering by topic")
def group_alignment_by_level(_alignment_data: pl.DataFrame, level: str) -> pl.DataFrame:
    """groups the data for the alignment chart by the level specified by the filters.
    Also calculates the fraction of total documents per type to visualise in the chart.

    Args:
        _alignment_data (pl.DataFrame): data for backend of alignment chart
        level (str): level to view, specified by domain/area filters

    Returns:
        pl.DataFrame: grouped alignment data for chart
    """
    total_pubs = _alignment_data.filter(
        pl.col("doc_type") == "publication_count"
    ).select(pl.sum("count"))
    total_patents = _alignment_data.filter(pl.col("doc_type") == "patent_count").select(
        pl.sum("count")
    )
    q = (
        _alignment_data.lazy()
        .with_columns(pl.col(level).cast(str))
        .groupby(["doc_type", level, "{}_name".format(level)])
        .agg([pl.sum("count").alias("total")])
        .with_columns(
            pl.when(pl.col("doc_type") == "publication_count")
            .then(pl.col("total") / total_pubs)
            .when(pl.col("doc_type") == "patent_count")
            .then(pl.col("total") / total_patents)
            .alias("doc_fraction")
        )
        .with_columns(
            pl.when(pl.col("doc_type") == "publication_count")
            .then("Publications")
            .when(pl.col("doc_type") == "patent_count")
            .then("Patents")
            .alias("doc_name_clean")
        )
        .with_columns((pl.col("doc_fraction") * 100).alias("doc_percentage"))
    )
    return q.collect()


@st.cache_data
def filter_novelty_by_level(
    _novelty_data: pl.DataFrame, _novelty_docs: pl.DataFrame, level: str, years: tuple
) -> pl.DataFrame:
    """Groups the data for the novelty chart by the level specified by the filters

    Args:
        _novelty_data (pl.DataFrame): data for backend of novelty chart
        level (str): level to view, specified by domain/area filters

    Returns:
        pl.DataFrame: novelty data for chart
    """
    _novelty_data = (
        _novelty_data.unique(subset=[level, "year"])
        .select(
            ["year"] + [col for col in _novelty_data.columns if col.startswith(level)]
        )
        .rename(
            {
                f"{level}_name": "name",
                f"{level}_doc_counts": "doc_counts",
                # f"{level}_novelty25": "novelty25",
                f"{level}_novelty50": "novelty50",
                # f"{level}_novelty75": "novelty75",
                f"{level}_entities": "entities",
            }
        )
    ).filter((pl.col("year") >= years[0]) & (pl.col("year") <= years[1]))

    # create dictionary of level ids as keys and names as values
    _topic_map = (
        _novelty_data.select([level, "name"])
        .unique()
        .to_pandas()
        .set_index(level)
        .to_dict()["name"]
    )

    # Create unique novelty_docs dataframe and add name column from _topic_map
    _novelty_docs = (
        _novelty_docs.unique(subset=[level, "work_id"])
        .select(["work_id", "year", "novelty", level])
        .filter((pl.col("year") >= years[0]) & (pl.col("year") <= years[1]))
        .with_columns(pl.col(level).cast(str).map_dict(_topic_map).alias("name"))
        .rename(
            {
                "work_id": "document_link",
                "year": "document_year",
                "name": "topic_name",
            }
        )
        .select(["document_link", "document_year", "topic_name", "novelty"])
    )

    return _novelty_data, _novelty_docs
    # map {level_name}


@st.cache_data
def group_filter_novelty_counts(
    _novelty_data: pl.DataFrame,
    _novelty_docs: pl.DataFrame,
    level: str,
    year_start: int,
    year_end: int,
) -> pl.DataFrame:
    """Groups the data for the novelty chart by the level specified by the filters

    Args:
        _novelty_data (pl.DataFrame): data for backend of novelty chart
        level (str): level to view, specified by domain/area filters
        year_start (int): start year for novelty chart
        year_end (int): end year for novelty chart

    Returns:
        pl.DataFrame: novelty data for chart
    """
    novelty_subdata = (
        _novelty_data.filter(
            (pl.col("year") >= year_start) & (pl.col("year") <= year_end)
        )
        .groupby(level)
        .agg(
            [
                pl.sum("doc_counts"),
                (pl.col("novelty50") * pl.col("doc_counts")).sum()
                / pl.col("doc_counts").sum(),
            ]
        )
        .join(_novelty_data.select([level, "name"]).unique(), on=level, how="left")
    )

    novelty_subdocs = (
        _novelty_docs.filter(
            (pl.col("year") >= year_start) & (pl.col("year") <= year_end)
        )
    ).sort(by="novelty", descending=True)

    return novelty_subdata, novelty_subdocs


@st.cache_data
def get_unique_words(series: pd.Series):
    return list(set(list(chain(*[x.split(" ") for x in series if isinstance(x, str)]))))


@st.cache_data
def get_ranked_novelty_articles(
    novelty_docs: pl.DataFrame, _doc_names: pl.DataFrame, topic: str, years: tuple
):
    novelty_docs = (
        novelty_docs.groupby(["document_link", "document_year", "novelty"])
        .agg({"topic_name": pl.list("topic_name")})
        .rename({"topic_name": "topic_names"})
    )
    if topic != "All":
        novelty_docs = novelty_docs.filter(
            novelty_docs["topic_names"].apply(lambda x: topic in x)
        )

    # add column with document_names by joining
    novelty_docs = novelty_docs.join(
        _doc_names.select(["work_id", "display_name"]),
        left_on="document_link",
        right_on="work_id",
        how="left",
    )[["document_link", "document_year", "display_name", "novelty", "topic_names"]]

    return novelty_docs


@st.cache_data
def filter_documents_with_entities(
    _novelty_docs: pl.DataFrame,
    _entity_dict: defaultdict,
    _doc_names: pl.DataFrame,
    entities: list,
    all_or_any: str = "All",
):

    # _entity_dict is a polars dataframe with entity and documents columns, the latter a list.
    if all_or_any == "Any":
        # Select rows with any of the entities in the list, that appaer in column entity
        document_ids = list(
            set(
                chain(
                    *[
                        _entity_dict.filter(pl.col("entity") == entity)["documents"]
                        for entity in entities
                        if entity in _entity_dict["entity"]
                    ]
                )
            )
        )

    else:
        # only return values in list that were found in all "entities" keys
        document_ids = list(
            chain(
                *[
                    _entity_dict.filter(pl.col("entity") == entity)[
                        "document"
                    ].to_list()
                    for entity in entities
                    if entity in _entity_dict["entity"]
                ]
            )
        )
        document_ids = list(set(document_ids[0]).intersection(*document_ids[1:]))

    # add "https://openalex.org/" prefix to each entity id
    document_ids = ["https://openalex.org/" + x for x in document_ids]

    _novelty_docs = _novelty_docs.filter(pl.col("document_link").is_in(document_ids))

    # add display_names
    _novelty_docs = _novelty_docs.join(
        _doc_names.select(["work_id", "display_name"]),
        left_on="document_link",
        right_on="work_id",
        how="left",
    )[["document_link", "document_year", "display_name", "novelty"]]

    return _novelty_docs
