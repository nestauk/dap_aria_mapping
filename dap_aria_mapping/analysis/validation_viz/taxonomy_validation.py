import streamlit as st
import streamlit_toggle as tog
import pandas as pd
from dap_aria_mapping.getters.validation import (
    get_chisq,
    get_entropy,
    get_tree_depths,
    get_pairwise_depth,
)
from dap_aria_mapping.getters.taxonomies import get_topic_names
import altair as alt
from collections import defaultdict
from itertools import chain
from dap_aria_mapping import PROJECT_DIR, taxonomy
from nesta_ds_utils.viz.altair import formatting
from streamlit_utils import *

formatting.setup_theme()
alt.data_transformers.disable_max_rows()


def validation_app():

    st.set_page_config(layout="wide")

    toggle_validation = tog.st_toggle_switch(
        "Display validation metrics",
        key="display_validation_metrics",
        default_value=False,
        label_after=True,
    )

    if toggle_validation:
        st.title("Metrics to Evaluate Various Taxonomies")
        algs = ["cooccur", "centroids", "imbalanced"]
        algs_alt = ["community", "centroids", "imbalanced"]

        st.header("Baseline taxonomies")
        dfs = get_taxonomy_data()

        (taxonomy_class, level_1, level_2, level_3, level_4) = st.columns(5)

        with taxonomy_class:
            st.subheader("Taxonomy")
            tax = st.selectbox(
                "Select a taxonomy",
                ["cooccur", "imbalanced", "centroids"],
            )

        dfs_tax = dfs[tax]
        levels_dict = {"L1": "All", "L2": "All", "L3": "All", "L4": "All"}

        with level_1:
            st.subheader("Level 1")
            l1 = st.selectbox(
                "Select a level 1 topic",
                ["All"] + sorted(list(dfs_tax["Level_1"].unique())),
            )

        if l1 == "All":
            level_shown = 1
            df_plot = dfs_tax
        else:
            levels_dict["L1"] = l1
            level_shown = 2
            df_plot = dfs_tax[dfs_tax["Level_1"] == l1]

            with level_2:
                st.subheader("Level 2")
                l2 = st.selectbox(
                    "Select a level 2 topic",
                    ["All"] + sorted(list(df_plot["Level_2"].unique())),
                )

            if l2 == "All":
                pass

            else:
                levels_dict["L2"] = l2
                level_shown = 3
                df_plot = df_plot[df_plot["Level_2"] == l2]

                with level_3:
                    st.subheader("Level 3")
                    l3 = st.selectbox(
                        "Select a level 3 topic",
                        ["All"] + sorted(list(df_plot["Level_3"].unique())),
                    )

                if l3 == "All":
                    pass
                else:
                    levels_dict["L3"] = l3
                    level_shown = 4
                    df_plot = df_plot[df_plot["Level_3"] == l3]

                    with level_4:
                        st.subheader("Level 4")
                        l4 = st.selectbox(
                            "Select a level 4 topic",
                            ["All"] + sorted(list(df_plot["Level_4"].unique())),
                        )

                    if l4 == "All":
                        pass
                    else:
                        levels_dict["L4"] = l4
                        level_shown = 5
                        df_plot = df_plot[df_plot["Level_4"] == l4]

        st.altair_chart(
            alt.Chart(df_plot)
            .mark_bar()
            .encode(
                x=alt.X(f"Level_{level_shown}:N",
                        axis=alt.Axis(title=""), sort="-y"),
                y=alt.Y("count()", axis=alt.Axis(title="", grid=False)),
                tooltip=[
                    alt.Tooltip(
                        f"Level_{str(level_shown)}_Entity_Names", title="Entities"
                    )
                ],
            )
            .properties(height=300, width=1200)
        )

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
            level_split = {alg: get_tree_depths(
                alg)["split_fractions"] for alg in algs}
            level_split_df = pd.DataFrame(level_split).unstack().reset_index()
            level_split_df.columns = ["algorithm", "level", "split_percentage"]

            split_chart = (
                alt.Chart(level_split_df)
                .mark_bar()
                .encode(
                    x=alt.X("algorithm:N", title=None),
                    y=alt.Y(
                        "split_percentage:Q", title=None, axis=alt.Axis(format="%")
                    ),
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
            topic_depth = {alg: get_pairwise_depth(
                alg, "topic") for alg in algs}
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
            subtopic_depth = {alg: get_pairwise_depth(
                alg, "subtopic") for alg in algs}
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
                subtopic_df = subtopic_df.loc[
                    subtopic_df["domain"].isin(domain_selections)
                ]

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
                df = df.loc[df.TopicName.str.contains(
                    "|".join(entity), case=False)]

            st.altair_chart(
                alt.Chart(df, width=180)
                .mark_circle(opacity=0.6)
                .encode(
                    x=alt.X(
                        "jitter:Q",
                        title=None,
                        axis=alt.Axis(values=[0], ticks=True,
                                      grid=False, labels=False),
                    ),
                    y=alt.Y(
                        "PropEntities",
                        axis=alt.Axis(title=""),
                        scale=alt.Scale(type="log"),
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
                        width=int(len(levels))
                        * int(len(taxonomy))
                        * int(len(ratio))
                        * 120,
                    )
                    .configure_axis(gridOpacity=0.3),
                )
            else:
                df_ts = get_timeseries_data(
                    df, taxonomies=taxonomy, ratios=ratio)

                chart_mean = (
                    alt.Chart(df_ts)
                    .mark_line(size=4)
                    .encode(
                        x=alt.X("level", title=None),
                        y=alt.Y(
                            "mean",
                            title=None,
                            scale=alt.Scale(
                                domain=(int(df_ts.lci.min()),
                                        int(df_ts.uci.max()))
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
                                (df_topic["topic"] == topic)
                                & (df_topic["taxonomy"] == alg)
                            ]
                        )
                        .mark_bar()
                        .encode(
                            x=alt.X("level:N", axis=alt.Axis(title="")),
                            y=alt.Y(
                                "prop:Q",
                                axis=alt.Axis(title="", grid=False,
                                              labelFontSize=8),
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
                                header=alt.Header(
                                    title=topic, labelFontSize=10),
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
                                    axis=alt.Axis(
                                        title="", grid=False, labelFontSize=8
                                    ),
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
                                    header=alt.Header(
                                        title=subtopic, labelFontSize=10),
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
                        *[alt.hconcat(*charts_topic)
                          for charts_topic in charts_],
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
                options=count_hists[alg_selections][top_level_selections].keys(
                ),
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
                options=frequency_hists[alg_selections][top_level_selections].keys(
                ),
                index=0,
            )

            st.components.v1.html(
                html=frequency_hists[alg_selections][top_level_selections][
                    next_level_selections
                ],
                height=600,
                scrolling=True,
            )

    st.header("ChatGPT topic naming")
    st.write(
        "The following table shows the topic naming for the ChatGPT model. \n"
        "The query is the following: \n\n"
        f"What are the Wikipedia topics that best describe the following groups of entities (the number of times in parenthesis corresponds to how often these entities are found in the topic, and should be taken into account when making a decision)?"
        " \n\n "
        'List1: "E: Acid (36194 times), Ion (22892 times), Oxygen (21824 times), Carbon (20138 times), X-ray crystallography (20066 times), Laser (18836 times), Spectroscopy (17298 times), Polymer (15761 times), PH (15166 times), Finite element method (14090 times), Correlation and dependence (13710 times), Nuclear magnetic resonance (13470 times), Electronvolt (12836 times), John Wiley & Sons (12460 times), Mass spectrometry (12429 times), Wavelength (11058 times), Ligand (10743 times), Microscopy (10490 times), Hydrogen (10151 times), Nitrogen (10125 times), Magnetic field (9798 times), Electrode (9303 times), Adsorption (9108 times), Scanning electron microscope (8961 times), Transmission electron microscopy (8540 times), Fluorescence (8429 times), Carboxylic acid (7584 times), Redox (7283 times), Renewable energy (7259 times), Molecular dynamics (7123 times), Viscosity (7067 times), Dielectric (6567 times), Graphene (6299 times), Photon (6275 times), Resonance (5953 times).'
        " \n\n "
        'List2: "E: Africa (18448 times), Australia (14272 times), India (13127 times), Brazil (8410 times), Italy (7977 times), South Africa (7925 times), Japan (6745 times), Kenya (4748 times), Russia (4319 times), Tanzania (3992 times), Mexico (3766 times), Uganda (3749 times), Bangladesh (3625 times), Pakistan (3614 times), Turkey (3608 times), Ghana (3358 times), Indonesia (3182 times), Hong Kong (2989 times), Portugal (2978 times), Malaysia (2928 times), Thailand (2839 times), Iran (2799 times), Saudi Arabia (2654 times), Ethiopia (2643 times), Malawi (2630 times), California (2523 times), Caribbean (2491 times), Chile (2446 times), South Korea (2404 times), Singapore (2402 times), Nepal (2386 times), Taiwan (2335 times), Greece (2205 times), Vietnam (2082 times), Argentina (2072 times)"'
        " \n\n "
        "Please only provide the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, please provide a list of entities that, if discarded, would help identify a topic. The structure of the answer should be a list of tuples of four elements: (list identifier, topic name, confidence score, list of entities to discard (None if there are none)). For example:"
        " \n\n"
        "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]"
        " \n\n"
        "Please avoid very general topic names (such as 'Science' or 'Technology') and return only the list of tuples with the answers using the structure above (it will be parsed by Python's ast literal_eval method)."
    )

    st.write(
        "Note that on follow up calls, it passes the following query:\n\n"
        f" Can you do the same to the following list of additional groups: \n\n <lists> \n\n"
        "Please only provide the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, please provide a list of entities that, if discarded, would help identify a topic. The structure of the answer should be a list of tuples of four elements: (list identifier, topic name, confidence score, list of entities to discard (None if there are none)). For example:"
        "\n\n"
        "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]"
    )

    (level_selector,) = st.columns(1)

    with level_selector:
        level_choice = st.selectbox(
            "Select the level to show",
            options=[1, 2, 3, 4],
            index=0,
            key="level_selector",
        )

    new_entity_names = get_topic_names(
        taxonomy_class="cooccur",
        name_type="entity",
        level=level_choice,
        n_top=35,
    )

    new_topic_names = get_topic_names(
        taxonomy_class="cooccur",
        name_type="chatgpt",
        level=level_choice,
        n_top=35,
    )

    new_entity_names = {
        k: v for k, v in new_entity_names.items() if k in new_topic_names.keys()
    }
    new_names_df = pd.DataFrame.from_dict(new_entity_names, orient="index").rename(
        columns={0: "entity_names"}
    )
    new_names_df = new_names_df.merge(
        pd.DataFrame.from_dict(new_topic_names, orient="index"),
        left_index=True,
        right_index=True,
    )
    st.write(new_names_df)


validation_app()
