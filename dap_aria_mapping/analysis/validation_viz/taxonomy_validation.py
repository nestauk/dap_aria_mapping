import streamlit as st
import streamlit_toggle as tog
import pandas as pd
from dap_aria_mapping.getters.validation import (
    get_chisq,
    get_entropy,
    get_tree_depths,
    get_pairwise_depth,
)
import altair as alt
from collections import defaultdict
from itertools import chain
from dap_aria_mapping import PROJECT_DIR, taxonomy
from nesta_ds_utils.viz.altair import formatting
from streamlit_utils import *

formatting.setup_theme()
alt.data_transformers.disable_max_rows()


def rec_dd():
    return defaultdict(rec_dd)


def validation_app():

    st.set_page_config(layout="wide")
    st.title("Metrics to Evaluate Various Taxonomies")
    algs = ["cooccur", "centroids", "imbalanced"]
    algs_alt = ["community", "centroids", "imbalanced"]

    st.header("Distribution Metrics")

    with st.expander(label="Number of Topics per Level of the Taxonomy"):
        level_size = {alg: get_tree_depths(alg)["counts"] for alg in algs}
        level_size_df = pd.DataFrame(level_size).unstack().reset_index()
        level_size_df.columns = ["algorithm", "level", "n_topics"]

        size_bars = (
            alt.Chart(level_size_df)
            .mark_bar()
            .encode(
                x=alt.X("algorithm:N", title=None, sort="-y"),
                y=alt.Y("n_topics:Q", title=None),
                color="algorithm:N",
            )
            .properties(width=200)
        )

        size_text = size_bars.mark_text(
            align="center",
            baseline="top",
            dy=-20,  # Nudges text to right so it doesn't appear on top of the bar
            fontSize=16,
        ).encode(text="n_topics:Q")
        size_chart = alt.layer(size_bars, size_text, data=level_size_df).facet(
            column="level:N"
        )

        st.altair_chart(size_chart)

    with st.expander(label="Percent of Topics that Split at the Next Level"):
        level_split = {alg: get_tree_depths(alg)["split_fractions"] for alg in algs}
        level_split_df = pd.DataFrame(level_split).unstack().reset_index()
        level_split_df.columns = ["algorithm", "level", "split_percentage"]

        split_chart = (
            alt.Chart(level_split_df)
            .mark_bar()
            .encode(
                x=alt.X("algorithm:N", title=None),
                y=alt.Y("split_percentage:Q", title=None, axis=alt.Axis(format="%")),
                color="algorithm:N",
                column=alt.Column("level:N", title="level"),
            )
            .properties(width=300)
        )

        st.altair_chart(split_chart)

    with st.expander(label="Entropy of Distributions of Entities per Topic"):
        entropy_data = {alg: get_entropy(alg) for alg in algs}
        entropy_data_df = pd.DataFrame(entropy_data).unstack().reset_index()
        entropy_data_df.columns = ["algorithm", "level", "entropy"]

        entropy_chart = (
            alt.Chart(entropy_data_df)
            .mark_bar()
            .encode(
                x=alt.X("algorithm:N", title=None),
                y=alt.Y("entropy:Q", title=None),
                color="algorithm:N",
                column=alt.Column("level:N", title=None),
            )
            .properties(width=200)
        )
        st.altair_chart(entropy_chart)

    with st.expander(
        label="Chi Square Test Statistic of Distributions of Entities per Topic"
    ):
        chisq_data = {alg: get_chisq(alg) for alg in algs}
        chisq_data_df = pd.DataFrame(chisq_data).unstack().reset_index()
        chisq_data_df.columns = ["algorithm", "level", "chi_sq"]

        chisq_chart = (
            alt.Chart(chisq_data_df)
            .mark_bar()
            .encode(
                x=alt.X("algorithm:N", title=None),
                y=alt.Y("chi_sq:Q", title=None),
                color="algorithm:N",
                column=alt.Column("level:N", title=None),
            )
            .properties(width=200)
        )
        st.altair_chart(chisq_chart)

    st.header("Content Metrics")

    with st.expander(
        label="Depth of Pairs of Topics Known to be Related Within Each Discipline"
    ):
        topic_depth = {alg: get_pairwise_depth(alg, "topic") for alg in algs}
        topic_depth_df = pd.DataFrame(topic_depth).unstack().reset_index()
        topic_depth_df = pd.concat(
            [topic_depth_df, topic_depth_df[0].apply(pd.Series)], axis=1
        )
        topic_depth_df = (
            topic_depth_df.drop(columns=0)
            .set_index(["level_0", "level_1"])
            .stack()
            .reset_index()
        )
        topic_depth_df.columns = [
            "algorithm",
            "domain",
            "discipline",
            "average pairwise depth",
        ]

        domain_filter = topic_depth_df["domain"].unique()
        domain_selections = st.multiselect(
            "Choose A Domain, leave blank to view all Domains", domain_filter
        )

        if len(domain_selections) > 0:
            topic_depth_df = topic_depth_df.loc[
                topic_depth_df["domain"].isin(domain_selections)
            ]

        topic_chart = (
            alt.Chart(topic_depth_df)
            .mark_bar()
            .encode(
                x=alt.X("algorithm:N", title=None),
                y=alt.Y("average pairwise depth:Q", title=None),
                color="algorithm:N",
                column=alt.Column(
                    "discipline:N", title=None, header=alt.Header(labelFontSize=9)
                ),
            )
            .properties(width=75)
        )
        st.altair_chart(topic_chart)

    with st.expander(
        label="Depth of Pairs of Subtopics Known to be Related Within Each Topic"
    ):
        subtopic_depth = {alg: get_pairwise_depth(alg, "subtopic") for alg in algs}
        temp = []
        for alg, alg_data in subtopic_depth.items():
            for domain, domain_data in alg_data.items():
                for discipline, discipline_data in domain_data.items():
                    for topic, score in discipline_data.items():
                        temp.append([alg, domain, discipline, topic, score])
        subtopic_df = pd.DataFrame(
            temp,
            columns=[
                "algorithm",
                "domain",
                "discipline",
                "topic",
                "average pairwise depth",
            ],
        )

        domain_filter = subtopic_df["domain"].unique()
        domain_selections = st.multiselect(
            "Choose A Domain, leave blank to view all Domains", domain_filter
        )

        discipline_filter = subtopic_df["discipline"].unique()
        discipline_selections = st.multiselect(
            "Choose A Discipline, leave blank to view all Disciplines",
            discipline_filter,
        )

        if len(domain_selections) > 0:
            subtopic_df = subtopic_df.loc[subtopic_df["domain"].isin(domain_selections)]

        if len(discipline_selections) > 0:
            subtopic_df = subtopic_df.loc[
                subtopic_df["discipline"].isin(discipline_selections)
            ]

        subtopic_chart = (
            alt.Chart(subtopic_df)
            .mark_bar()
            .encode(
                x=alt.X("algorithm:N", title=None),
                y=alt.Y("average pairwise depth:Q", title=None),
                color="algorithm:N",
                column=alt.Column(
                    "topic:N", title=None, header=alt.Header(labelFontSize=12)
                ),
            )
            .properties(width=150)
        )
        st.altair_chart(subtopic_chart)

    st.header("Simulation Metrics")

    (simulation_select,) = st.columns(1)

    with simulation_select:
        simulation = st.selectbox(
            "Simulation Type",
            options=[
                "noise",
                "sample",
            ],
            index=0,
        )

    with st.form(key="kello"):

        (
            level_select,
            taxonomy_select,
            ratio_select,
            entity_select,
        ) = st.columns(4)

        with level_select:
            levels = st.multiselect(
                "Taxonomy Levels",
                options=[str(x) for x in range(1, 6)],
                default=[str(x) for x in range(1, 6)],
            )

        with taxonomy_select:
            taxonomy = st.multiselect(
                "Taxonomy Class", options=algs_alt, default=["community"]
            )

        with ratio_select:
            if simulation == "noise":
                ratio = st.multiselect(
                    "Noise Ratio",
                    options=[
                        "00",
                        "05",
                        "07",
                        "08",
                        "11",
                        "14",
                        "18",
                        "24",
                        "31",
                        "39",
                        "51",
                    ],
                    default=["00"],
                )
            elif simulation == "sample":
                ratio = st.multiselect(
                    "Sample Ratio",
                    options=[
                        "10",
                        "20",
                        "30",
                        "40",
                        "50",
                        "60",
                        "70",
                        "80",
                        "90",
                        "100",
                    ],
                    default=["100"],
                )
        submit_button = st.form_submit_button(label="Submit")

    with st.expander(label="Entropy of Distributions of Entities per Topic"):
        df = get_queried_sizes(
            taxonomies=taxonomy, ratios=ratio, levels=levels, simulation=simulation
        )

        with entity_select:
            options = list(
                set(
                    list(
                        chain(
                            *[
                                str(x).split(", ")
                                for x in df["TopicName"].str.lstrip("E: ").unique()
                            ]
                        )
                    )
                )
            )
            entity = st.multiselect(
                "Select relevant entities",
                options=["All entities"] + options,
                default=["All entities"],
            )
            if "All entities" in entity:
                entity = "All entities"

        if entity != "All entities":
            df = df.loc[df.TopicName.str.contains("|".join(entity), case=False)]

        st.altair_chart(
            alt.Chart(df, width=180)
            .mark_circle(opacity=0.6)
            .encode(
                x=alt.X(
                    "jitter:Q",
                    title=None,
                    axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
                ),
                y=alt.Y(
                    "PropEntities", axis=alt.Axis(title=""), scale=alt.Scale(type="log")
                ),
                size=alt.Size(
                    "PropEntities", scale=alt.Scale(range=[12, 1000]), legend=None
                ),
                color=alt.Color("parent_topic:N", legend=None),
                tooltip=["Topic", "TopicName", "Proportion"],
                column=alt.Column(
                    "level:N",
                    header=alt.Header(
                        labelAngle=-90,
                        titleOrient="top",
                        labelOrient="bottom",
                        labelAlign="right",
                    ),
                ),
            )
            .transform_calculate(jitter="sqrt(-2*log(random()))*cos(2*PI*random())")
            .configure_facet(spacing=0)
            .configure_axis(gridOpacity=0.1)
            .configure_view(stroke=None)
            .properties(title="Topic sizes")
        )

        display_size_data = st.checkbox("Show Size Data")
        if display_size_data:
            st.write(df)

    with st.expander(label="Topic Distance to Centroid"):
        toggle = tog.st_toggle_switch(
            "Display as time series",
            key="display_time_series",
            default_value=False,
            label_after=True,
        )

        df = get_queried_distances(
            taxonomies=taxonomy,
            ratios=ratio,
            levels=levels,
            simulation=simulation,
            entities=entity,
        )

        if not toggle:
            st.altair_chart(
                alt.Chart(df)
                .mark_boxplot(extent="min-max")
                .encode(
                    x=alt.X("level:N", axis=alt.Axis(title="")),
                    y=alt.Y("distance:Q", axis=alt.Axis(title="")),
                    color=alt.Color("taxonomy:N", legend=None),
                    tooltip=["level", "distance"],
                )
                .properties(
                    title="Distance to topic centroids",
                    height=300,
                    width=int(len(levels)) * int(len(taxonomy)) * int(len(ratio)) * 120,
                )
                .configure_axis(gridOpacity=0.3),
            )
        else:
            df_ts = get_timeseries_data(df, taxonomies=taxonomy, ratios=ratio)

            chart_mean = (
                alt.Chart(df_ts)
                .mark_line(size=4)
                .encode(
                    x=alt.X("level", title=None),
                    y=alt.Y(
                        "mean",
                        title=None,
                        scale=alt.Scale(
                            domain=(int(df_ts.lci.min()), int(df_ts.uci.max()))
                        ),
                    ),
                    color=alt.Color("config", title=None),
                )
                .properties(width=1280, height=480)
            )

            chart_band = (
                alt.Chart(df_ts)
                .mark_area(opacity=0.1)
                .encode(
                    x=alt.X("level", title=None),
                    y="lci",
                    y2="uci",
                    color=alt.Color("config", title=None),
                )
                .properties(width=1280, height=480)
            )

            st.altair_chart(
                alt.layer(chart_mean, chart_band).properties(
                    title="Distance to topic centroids",
                )
            )

        display_distance_data = st.checkbox("Show Distance data")
        if display_distance_data:
            st.write(df)

    with st.expander(label="Pairwise Topic Plots"):
        topic_selection = st.selectbox(
            label="Select a topic class", options=["topic", "subtopic"]
        )

        df_topic, df_subtopic = get_pairwise_data(
            taxonomies=taxonomy,
            ratios=ratio,
            simulation=simulation,
            topic_selection=topic_selection,
        )

        if topic_selection == "topic":
            nested_charts = [
                [
                    alt.Chart(
                        df_topic.loc[
                            (df_topic["topic"] == topic) & (df_topic["taxonomy"] == alg)
                        ]
                    )
                    .mark_bar()
                    .encode(
                        x=alt.X("level:N", axis=alt.Axis(title="")),
                        y=alt.Y(
                            "prop:Q",
                            axis=alt.Axis(title="", grid=False, labelFontSize=8),
                            scale=alt.Scale(domain=[0, 1]),
                        ),
                        row=alt.Row(
                            "ratio:N",
                            header=alt.Header(
                                title="Ratio", labelFontSize=10, titleFontSize=8
                            ),
                        ),
                        column=alt.Column(
                            "subtopic:N",
                            header=alt.Header(title=topic, labelFontSize=10),
                        ),
                        tooltip=["pairs"],
                    )
                    .properties(height=160, width=120)
                    for topic in df_topic.topic.unique()
                ]
                for alg in taxonomy
            ]

            charts_concat = []
            for i, alg in enumerate(taxonomy):
                charts_ = nested_charts[i]
                charts_concat_ = alt.hconcat(
                    *charts_, title=f"Pairwise combinations - {alg}"
                )
                charts_concat.append(charts_concat_)

            st.altair_chart(alt.vconcat(*charts_concat))

        else:
            nested_charts = [
                [
                    [
                        alt.Chart(
                            df_subtopic.loc[
                                (df_subtopic["subtopic"] == subtopic)
                                & (df_subtopic["taxonomy"] == alg)
                            ]
                        )
                        .mark_bar()
                        .encode(
                            x=alt.X("level:N", axis=alt.Axis(title="")),
                            y=alt.Y(
                                "prop:Q",
                                axis=alt.Axis(title="", grid=False, labelFontSize=8),
                                scale=alt.Scale(domain=[0, 1]),
                            ),
                            # color="ratio:N",
                            row=alt.Row(
                                "ratio:N",
                                header=alt.Header(
                                    title="Ratio", labelFontSize=10, titleFontSize=8
                                ),
                            ),
                            column=alt.Column(
                                "concept:N",
                                header=alt.Header(title=subtopic, labelFontSize=10),
                            ),
                            tooltip=["pairs"],
                        )
                        .properties(height=160, width=120)
                        for subtopic in df_subtopic.loc[
                            df_subtopic.topic == topic
                        ].subtopic.unique()
                    ]
                    for topic in df_subtopic.topic.unique()
                ]
                for alg in taxonomy
            ]

            charts_concat = []
            for i, alg in enumerate(taxonomy):
                charts_ = nested_charts[i]
                charts_concat_ = alt.hconcat(
                    *[alt.hconcat(*charts_topic) for charts_topic in charts_],
                    title=f"Pairwise combinations - {alg}",
                )
                charts_concat.append(charts_concat_)

            st.altair_chart(alt.vconcat(*charts_concat))

    with st.expander("Number of Topics Present Per Journal"):
        count_hists = get_count_histograms()

        alg_selections = st.selectbox(
            label="choose an algorithm", options=count_hists.keys(), index=0
        )
        top_level_selections = st.selectbox(
            label="choose the top level",
            options=sorted(list(count_hists[alg_selections].keys())),
            index=0,
        )
        next_level_selections = st.selectbox(
            label="choose the next level",
            options=count_hists[alg_selections][top_level_selections].keys(),
            index=0,
        )

        st.components.v1.html(
            html=count_hists[alg_selections][top_level_selections][
                next_level_selections
            ],
            height=600,
            scrolling=True,
        )

    with st.expander("Percentage Breakdown of Topics per Journal"):
        frequency_hists = get_frequency_histograms()

        alg_selections = st.selectbox(
            label="choose an algorithm (percentage breakdown)",
            options=frequency_hists.keys(),
            index=0,
        )
        top_level_selections = st.selectbox(
            label="choose the top level (percentage breakdown)",
            options=sorted(list(frequency_hists[alg_selections].keys())),
            index=0,
        )
        next_level_selections = st.selectbox(
            label="choose the next level (percentage breakdown)",
            options=frequency_hists[alg_selections][top_level_selections].keys(),
            index=0,
        )

        st.components.v1.html(
            html=frequency_hists[alg_selections][top_level_selections][
                next_level_selections
            ],
            height=600,
            scrolling=True,
        )


validation_app()
