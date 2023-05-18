import streamlit as st
from PIL import Image
import altair as alt
from nesta_ds_utils.viz.altair import formatting
from dap_aria_mapping import PROJECT_DIR, IMAGE_DIR
from dap_aria_mapping.getters.app_tables.horizon_scanner import (
    volume_per_year,
    novelty_per_year,
    novelty_documents,
    documents_with_entities,
    get_entities,
)

from dap_aria_mapping.utils.app_utils import convert_to_pandas
from dap_aria_mapping.getters.taxonomies import get_topic_names, get_entity_embeddings
import polars as pl
import pandas as pd
import numpy as np
from typing import List, Tuple
from itertools import chain

formatting.setup_theme()

PAGE_TITLE = "Horizon Scanner"

# icon to be used as the favicon on the browser tab
icon = Image.open(f"{IMAGE_DIR}/hs_icon.ico")

# sets page configuration with favicon and title
st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon=icon)


@st.cache_data(show_spinner="Loading data")
def load_overview_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str]]:
    """loads in the volume per year chart and does initial formatting that is not impacted by filters
    caches results so the data is not loaded each time a filter is run

    Returns:
        pl.DataFrame: total patents/publications per domain/area/topic per year, with names
        pl.DataFrame: same as above, but patent/publication counts are melted to long form
        List: unique domain names in dataset
    """
    loading_bar = st.progress(0, "Loading data")
    volume_data = volume_per_year()

    loading_bar.progress((1/6)*1, text="Loading data")
    # generate a list of the unique domain names to use as the filter
    unique_domains = volume_data["domain_name"].unique().to_list()
    unique_domains.insert(0, "All")
    loading_bar.progress((1/6)*2, text="Loading data")
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
    loading_bar.progress((1/6)*3, text="Loading data")

    novelty_data = novelty_per_year()
    loading_bar.progress((1/6)*4, text="Loading data")
    novelty_docs = novelty_documents()
    loading_bar.progress((1/6)*5, text="Loading data")
    docs_with_entities = documents_with_entities()
    loading_bar.progress((1/6)*6, text="Loading data")

    return (
        volume_data,
        alignment_data,
        novelty_data,
        novelty_docs,
        unique_domains,
        docs_with_entities,
    )


@st.cache_data(show_spinner="Filtering by domain")
def filter_by_domain(
    domain: str,
    _volume_data: pl.DataFrame,
    _alignment_data: pl.DataFrame,
    _novelty_data: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str]]:
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
    alignment_data = _alignment_data.filter(pl.col("domain_name") == domain)
    novelty_data = _novelty_data.filter(pl.col("domain_name") == domain)
    unique_areas = volume_data["area_name"].unique().to_list()
    return volume_data, alignment_data, novelty_data, unique_areas


@st.cache_data(show_spinner="Filtering by area")
def filter_by_area(
    area: str,
    _volume_data: pl.DataFrame,
    _alignment_data: pl.DataFrame,
    _novelty_data: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str]]:
    """filters volume data, alignment data, and filter options based on an area selection

    Args:
        area (str): domain selected by the filter
        _volume_data (pl.DataFrame): volume data for emergence chart
        _alignment_data (pl.DataFrame): alignment data for alignment chart
        _novelty_data (pl.DataFrame): novelty data for novelty chart

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str]]: updated dataframes filtered by an area, and a list of unique topics to populate topic filter
    """
    volume_data = _volume_data.filter(pl.col("area_name") == area)
    alignment_data = _alignment_data.filter(pl.col("area_name") == area)
    novelty_data = _novelty_data.filter(pl.col("area_name") == area)
    unique_topics = volume_data["topic_name"].unique().to_list()
    return volume_data, alignment_data, novelty_data, unique_topics


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


st.cache_data(show_spinner="Filtering by topic")


@st.cache_data
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
                f"{level}_novelty25": "novelty25",
                f"{level}_novelty50": "novelty50",
                f"{level}_novelty75": "novelty75",
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
        .select(["work_id", "display_name", "year", "novelty", level])
        .filter((pl.col("year") >= years[0]) & (pl.col("year") <= years[1]))
        .with_columns(pl.col(level).cast(str).map_dict(_topic_map).alias("name"))
        .rename(
            {
                "work_id": "document_link",
                "display_name": "document_name",
                "year": "document_year",
                "name": "topic_name",
            }
        )
        .select(
            ["document_link", "document_name", "document_year", "topic_name", "novelty"]
        )
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
def filter_documents_with_entities(
    _novelty_docs: pl.DataFrame, docs_with_entities: pl.DataFrame, entities: list
):
    # filter docs_with_entities to only keep documents with entities in entities
    docs_with_entities = docs_with_entities.filter(pl.col("entity_list").isin(entities))

    # join novelty_docs and docs_with_entities only where there's match
    docs_with_entities = _novelty_docs.join(
        docs_with_entities, left_on="work_id", right_on="document_id", how="inner"
    )
    return docs_with_entities


header1, header2 = st.columns([1, 10])
with header1:
    st.image(icon)
with header2:
    st.markdown(
        f'<h1 style="color:#0000FF;font-size:72px;">{"Horizon Scanner"}</h1>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<h1 style="color:#0000FF;font-size:16px;">{"<em>Explore patterns and trends in research domains across the UK<em>"}</h1>',
    unsafe_allow_html=True,
)

# load in volume, alignment, and novelty data
(
    volume_data,
    alignment_data,
    novelty_data,
    novelty_docs,
    unique_domains,
    docs_with_entities,
) = load_overview_data()

with st.sidebar:
    # filter for domains comes from unique domain names
    domain = st.selectbox(label="Select a Domain", options=unique_domains)
    area = "All"
    topic = "All"
    level_considered = "domain"
    # if a domain is selected in the filter, then filter the data
    if domain != "All":
        volume_data, alignment_data, novelty_data, unique_areas = filter_by_domain(
            domain, volume_data, alignment_data, novelty_data
        )
        unique_areas.insert(0, "All")
        # if a domain is selected, the plots that are being visualised are by area (i.e. level 2)
        level_considered = "area"

        # if a domain is selected, allow user to filter by area
        area = st.selectbox(label="Select an Area", options=unique_areas)
        if area != "All":
            # if an area is selected, filter data to the area and present at topic level
            volume_data, alignment_data, novelty_data, unique_topics = filter_by_area(
                area, volume_data, alignment_data, novelty_data
            )
            level_considered = "topic"

    st.sidebar.markdown("---")
    years = st.slider("Select a range of years", 2007, 2022, (2007, 2022))


overview_tab, disruption_tab, novelty_tab, overlaps_tab = st.tabs(
    ["Overview", "Disruption", "Novelty", "Overlaps"]
)
with overview_tab:

    st.subheader("Growth Over Time")
    st.markdown(
        "View trends in volume of content over time to detect emerging or stagnant areas of innovation"
    )
    show_only = st.selectbox(
        label="Show Emergence In:", options=["All Documents", "Publications", "Patents"]
    )
    if show_only == "Publications":
        y_col = "publication_count"
    elif show_only == "Patents":
        y_col = "patent_count"
    else:
        y_col = "total_docs"

    emergence_data = convert_to_pandas(
        group_emergence_by_level(volume_data, level_considered, y_col)
    )

    volume_chart = (
        alt.Chart(emergence_data)
        .mark_line(point=True)
        .encode(
            alt.X("year:N"),
            alt.Y("{}:Q".format(y_col), title="Total Documents"),
            color=alt.Color(
                "{}_name:N".format(level_considered),
                legend=alt.Legend(
                    labelFontSize=10, title=None, labelLimit=0, symbolSize=20
                ),
            ),
            tooltip=[
                alt.Tooltip("year:N", title="Year"),
                alt.Tooltip("{}:Q".format(y_col), title="Total Documents"),
                alt.Tooltip(
                    "{}_name:N".format(level_considered),
                    title="{}".format(level_considered),
                ),
            ],
        )
        .interactive()
        .properties(width=1100, height=500)
    )
    st.altair_chart(volume_chart)

    st.subheader("Alignment in Research and Industry")
    st.markdown(
        "Areas with high publication count and low patent count indicates there is significantly more activity in academia than industry on this topic (or vice versa)."
    )
    filtered_alignment_data = convert_to_pandas(
        group_alignment_by_level(alignment_data, level_considered)
    )
    alignment_chart = (
        alt.Chart(filtered_alignment_data)
        .transform_filter(alt.datum.doc_fraction > 0)
        .mark_point(size=60)
        .encode(
            alt.X(
                "doc_fraction:Q",
                title="Percent of Documents of the Given Type",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(tickSize=0, format="%", grid=False),
            ),
            alt.Y(
                "{}_name:N".format(level_considered),
                axis=alt.Axis(labelLimit=0, title=None, grid=True),
            ),
            tooltip=[
                alt.Tooltip("doc_name_clean:N", title="Document Type"),
                alt.Tooltip(
                    "doc_percentage:Q", format=".2f", title="Percent of Docs (%)"
                ),
                alt.Tooltip(
                    "{}_name:N".format(level_considered),
                    title="{}".format(level_considered),
                ),
            ],
            color=alt.Color(
                "doc_name_clean:N",
                legend=alt.Legend(
                    direction="horizontal",
                    legendX=10,
                    legendY=-80,
                    orient="none",
                    titleAnchor="middle",
                    title=None,
                ),
            ),
        )
        .interactive()
        .properties(width=1100)
    )

    st.altair_chart(alignment_chart)

with disruption_tab:
    disruption_trends, disruption_drilldown = st.columns(2)

    with disruption_trends:
        st.subheader("Trends in Disruption")
        st.markdown(
            "This could show if certain domains/areas have been recently disrupted or have lacked disruption"
        )

    with disruption_drilldown:
        st.subheader("Drill Down in Disruption")
        st.markdown(
            "This would allow a user to select a topic and see the distribution of disruptiveness of papers within that topic"
        )

with novelty_tab:

    novelty_charts_tab, novelty_docs_tab = st.tabs(["Charts", "Search"])
    with novelty_charts_tab:

        st.subheader("Trends in Novelty")
        st.markdown(
            "View trends in novelty of content over time to detect emerging or stagnant areas of innovation"
        )

        filtered_novelty_data, filtered_novelty_docs = filter_novelty_by_level(
            _novelty_data=novelty_data,
            _novelty_docs=novelty_docs,
            level=level_considered,
            years=years,
        )

        col1, col2 = st.columns([0.65, 0.35])
        with col1:

            selection = alt.selection_single(
                fields=["name"], on="click"
            )  # bind="legend")#nearest=True, on='mouseover', empty="none")
            novelty_bump_chart = (
                alt.Chart(convert_to_pandas(filtered_novelty_data))
                .mark_line(point=True)
                .encode(
                    x=alt.X("year:O"),
                    y="rank:Q",
                    color=alt.Color(
                        "name:N",
                        title=None,
                        legend=alt.Legend(
                            labelFontSize=10, title=None, labelLimit=0, symbolSize=20
                        ),
                    ),
                    tooltip=["name:N", "doc_counts:Q", "entities:N"],
                    opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
                )
                .transform_window(
                    rank="rank()",
                    sort=[alt.SortField("novelty50", order="ascending")],
                    groupby=["year"],
                )
                .properties(
                    height=150
                    * np.log(1 + filtered_novelty_data["name"].unique().shape[0]),
                    # width=900,
                    padding={"left": 50, "top": 10, "right": 10, "bottom": 50},
                )
                .add_selection(selection)
            )

            # selected_data = alt.data_transformers.get()(novelty_bump_chart.data)
            # st.session_state.selected_category = selected_data['category'][0]

            st.altair_chart(novelty_bump_chart, use_container_width=True)

        with col2:

            novelty_bubbles, novelty_docs = group_filter_novelty_counts(
                _novelty_data=filtered_novelty_data,
                _novelty_docs=novelty_docs,
                level=level_considered,
                year_start=years[0],
                year_end=years[1],
            )

            novelty_bubble_chart = (
                alt.Chart(convert_to_pandas(novelty_bubbles))
                .mark_circle()
                .encode(
                    x=alt.X(
                        "novelty50",
                        title="Novelty Score",
                        scale=alt.Scale(zero=False),
                    ),
                    y=alt.Y(
                        "name",
                        title="",
                        sort=alt.EncodingSortField(
                            field="novelty50", order="descending"
                        ),
                        axis=alt.Axis(labels=False, ticks=False, domain=False),
                    ),
                    size=alt.Size(
                        "doc_counts",
                        title="",
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("name", title="Topic title"),
                        alt.Tooltip(
                            "novelty50", title="Topic novelty (document-based)"
                        ),
                        alt.Tooltip("doc_counts", title="Number of documents"),
                    ],
                )
            )

            labels = novelty_bubble_chart.mark_text(
                align="left", baseline="middle", dx=7
            ).encode(
                # limit text length
                text=alt.Text("name:N"),
                size=alt.Size(),
            )

            st.altair_chart(novelty_bubble_chart + labels, use_container_width=True)

        # Display most novel articles
        st.subheader("Relevant Articles")
        filtered_novelty_docs_pd = convert_to_pandas(filtered_novelty_docs)

        # Keep topic_name non-None
        filtered_novelty_docs_pd = filtered_novelty_docs_pd[
            filtered_novelty_docs_pd["topic_name"].notnull()
        ]

        # Selectbox to allow user to select one among the df's many topic_names. Display only the corresponding topic_names.
        novelty_docs_topic = st.selectbox(
            "Select a topic to view the most novel articles",
            ["All"] + sorted(filtered_novelty_docs_pd["topic_name"].unique()),
        )

        # [TASK] These need to be turned into functions, otherwise they reload wahen searching.
        # Filter the df to display only the selected topic_name
        if novelty_docs_topic != "All":
            filtered_novelty_docs_pd = filtered_novelty_docs_pd[
                filtered_novelty_docs_pd["topic_name"] == novelty_docs_topic
            ]

        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.markdown("Most Novel Articles")
            st.dataframe(
                filtered_novelty_docs_pd.sort_values(
                    by=["novelty"], ascending=False
                ).head(50)
            )
        with col2:
            st.markdown("Least Novel Articles")
            st.dataframe(
                filtered_novelty_docs_pd.sort_values(
                    by=["novelty"], ascending=True
                ).head(50)
            )

    with novelty_docs_tab:

        unique_keywords = get_entities()

        st.title("Search Articles")
        with st.form("my_form"):
            col1, col2, col3 = st.columns([0.8, 0.1, 0.1])
            with col1:
                # query keywords, where each time
                query = st.multiselect("Keywords", unique_keywords)
            with col2:
                # ask for either all or at least one
                all_or_any = st.radio("Match all or any keywords?", ["All", "Any"])
            with col3:
                # number of results
                top_n = st.number_input(
                    "Number of results", min_value=1, max_value=100, value=10
                )

            submitted = st.form_submit_button("Submit")

            if submitted:
                if all_or_any == "All":
                    query_df = filter_documents_with_entities(
                        filtered_novelty_docs, docs_with_entities, query
                    )
                else:
                    query_df = filter_documents_with_entities(
                        filtered_novelty_docs, docs_with_entities, query
                    )

                query_df = query_df.sort_values(by="novelty", ascending=False).head(
                    top_n
                )
                st.dataframe(query_df)

        # with col4:
        #     if not query:
        #         matching_articles = 0
        #     else:
        #         matching_articles = len(_novelty_docs)
        #     st.markdown(f"Count: {matching_articles}")


with overlaps_tab:
    heatmap, overlap_drilldown = st.columns(2)
    with heatmap:
        st.subheader("Heatmap of Overlaps")
        st.markdown(
            "This would be used to show which areas have a large amount of research that spans multiple topics (i.e. a lot of research combining ML with Neuroscience)"
        )
    with overlap_drilldown:
        st.subheader("Trends in Overlaps")
        st.markdown(
            "This would allow a user to select interesting overlaps and view the trends in growth over time"
        )


# adds the nesta x aria logo at the bottom of each tab, 3 lines below the contents
st.markdown("")
st.markdown("")
st.markdown("")

white_space, logo, white_space = st.columns([1.5, 1, 1.5])
with logo:
    st.image(Image.open(f"{IMAGE_DIR}/igl_nesta_aria_logo.png"))
