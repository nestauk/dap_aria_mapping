import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import altair as alt
from nesta_ds_utils.viz.altair import formatting
from dap_aria_mapping import PROJECT_DIR, IMAGE_DIR
from dap_aria_mapping.getters.app_tables.horizon_scanner import (
    get_entities,
)

from dap_aria_mapping.analysis.app.pages.utils.horizon_scanner import *

from dap_aria_mapping.utils.app_utils import convert_to_pandas
import polars as pl
import numpy as np
from toolz import pipe

formatting.setup_theme()

PAGE_TITLE = "Horizon Scanner"

# icon to be used as the favicon on the browser tab
icon = Image.open(f"{IMAGE_DIR}/hs_icon.ico")

# sets page configuration with favicon and title
st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon=icon)


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
    unique_domains,
) = load_overview_data()

with st.sidebar:
    # filter for domains comes from unique domain names
    show_novelty = st.checkbox("Show Novelty")
    if show_novelty:
        (novelty_data, novelty_docs, document_names, entity_dict) = load_novelty_data()
    st.markdown("---")
    domain = st.selectbox(label="Select a Domain", options=unique_domains)
    area = "All"
    topic = "All"
    level_considered = "domain"
    # if a domain is selected in the filter, then filter the data
    if domain != "All":
        volume_data, alignment_data, unique_areas = filter_volume_by_domain(
            domain, volume_data, alignment_data
        )
        if show_novelty:
            novelty_data = filter_novelty_by_domain(domain, novelty_data)
        unique_areas.insert(0, "All")
        # if a domain is selected, the plots that are being visualised are by area (i.e. level 2)
        level_considered = "area"

        # if a domain is selected, allow user to filter by area
        area = st.selectbox(label="Select an Area", options=unique_areas)
        if area != "All":
            # if an area is selected, filter data to the area and present at topic level
            volume_data, alignment_data, unique_topics = filter_volume_by_area(
                area, volume_data, alignment_data
            )
            if show_novelty:
                novelty_data = filter_novelty_by_area(area, novelty_data)
            level_considered = "topic"

    st.sidebar.markdown("---")
    years = st.slider("Select a range of years", 2007, 2022, (2007, 2022))

if show_novelty:
    overview_tab, disruption_tab, novelty_tab, overlaps_tab = st.tabs(
        ["Overview", "Disruption", "Novelty", "Overlaps"]
    )
else:
    overview_tab, disruption_tab, overlaps_tab = st.tabs(
        ["Overview", "Disruption", "Overlaps"]
    )

# tabs = option_menu(
#     None,
#     ["Overview", "Disruption", "Novelty", "Overlaps"],
#     icons=["house", "cloud-upload", "list-task", "gear"],
#     menu_icon="cast",
#     default_index=0,
#     orientation="horizontal",
#     styles={
#         "container": {"padding": "0!important", "background-color": "#fafafa"},
#         "icon": {"color": "orange", "font-size": "25px"},
#         "nav-link": {
#             "font-size": "25px",
#             "text-align": "left",
#             "margin": "0px",
#             "--hover-color": "#eee",
#         },
#         "nav-link-selected": {"background-color": "green"},
#     },
# )

# if tabs == "Overview":
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

# if tabs == "Disruption":
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

# if tabs == "Novelty":
if show_novelty:
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
                                labelFontSize=10,
                                title=None,
                                labelLimit=0,
                                symbolSize=20,
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

                st.altair_chart(
                    novelty_bubble_chart + labels,
                    use_container_width=True,
                    height=150
                    * np.log(1 + filtered_novelty_data["name"].unique().shape[0]),
                )

            # Display most novel articles
            st.subheader("Relevant Articles")

            # Selectbox to allow user to select one among the df's many topic_names. Display only the corresponding topic_names.
            if domain == "All":
                topic_sel = unique_domains
            elif domain != "All" and area == "All":
                topic_sel = unique_areas
            elif domain != "All" and area != "All":
                topic_sel = unique_topics
            novelty_docs_topic = st.selectbox(
                "Select a topic to view the most and least novel articles",
                ["All"] + sorted(topic_sel, key=lambda x: x.lower()),
            )

            filtered_topic_novelty_docs = get_ranked_novelty_articles(
                _novelty_docs=filtered_novelty_docs,
                _doc_names=document_names,
                _topic=novelty_docs_topic,
            )
            col1, col2 = st.columns([0.5, 0.5])
            filtered_topic_novelty_docs = convert_to_pandas(filtered_topic_novelty_docs)

            with col1:
                st.markdown("Most Novel Articles")
                st.dataframe(
                    filtered_topic_novelty_docs.sort_values(
                        by=["novelty"], ascending=False
                    ).head(50)
                )
            with col2:
                st.markdown("Least Novel Articles")
                st.dataframe(
                    filtered_topic_novelty_docs.sort_values(
                        by=["novelty"], ascending=True
                    ).head(50)
                )

        with novelty_docs_tab:

            unique_keywords = pipe(get_entities(), lambda x: sorted(x))

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
                    query_df = filter_documents_with_entities(
                        _novelty_docs=filtered_novelty_docs,
                        _entity_dict=entity_dict,
                        _doc_names=document_names,
                        entities=query,
                        all_or_any=all_or_any,
                    )

                    query_df = convert_to_pandas(query_df)

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

    # if tabs == "Overlaps":
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
