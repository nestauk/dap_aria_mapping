import streamlit as st
import pandas as pd
from dap_aria_mapping.getters.validation import get_chisq, get_entropy, get_tree_depths, get_pairwise_depth
import altair as alt
from nesta_ds_utils.viz.altair import formatting
formatting.setup_theme()

def validation_app():
    st.set_page_config(layout="wide")
    st.title("Metrics to Evaluate Various Taxonomys")
    algs = ['cooccur','centroids', 'imbalanced']
    
    st.header("Distribution Metrics")

    with st.expander(label ="Number of Topics per Level of the Taxonomy"):
        level_size = {alg: get_tree_depths(alg)["counts"] for alg in algs}
        level_size_df = pd.DataFrame(level_size).unstack().reset_index()
        level_size_df.columns = ['algorithm', 'level', 'n_topics']

        size_bars = alt.Chart(level_size_df).mark_bar().encode(
            x=alt.X('algorithm:N',title=None, sort = "-y"),
            y=alt.Y('n_topics:Q',title=None),
            color='algorithm:N'
        ).properties(width=200)

        size_text = size_bars.mark_text(
            align='center',
            baseline='top',
            dy=-20, # Nudges text to right so it doesn't appear on top of the bar
            fontSize = 16).encode(
            text='n_topics:Q'
)       
        size_chart = alt.layer(size_bars, size_text, data=level_size_df).facet(column='level:N')

        st.altair_chart(size_chart)
    
    with st.expander(label ="Percent of Topics that Split at the Next Level"):
        level_split = {alg: get_tree_depths(alg)["split_fractions"] for alg in algs}
        level_split_df = pd.DataFrame(level_split).unstack().reset_index()
        level_split_df.columns = ['algorithm', 'level', 'split_percentage']

        split_chart = alt.Chart(level_split_df).mark_bar().encode(
            x=alt.X('algorithm:N',title=None),
            y=alt.Y('split_percentage:Q',title=None, axis=alt.Axis(format='%')),
            color='algorithm:N',
            column=alt.Column('level:N', title="level")
        ).properties(width=300)

        st.altair_chart(split_chart)

    with st.expander(label ="Entropy of Distributions of Entities per Topic"):
        entropy_data = {alg: get_entropy(alg) for alg in algs}
        entropy_data_df = pd.DataFrame(entropy_data).unstack().reset_index()
        entropy_data_df.columns = ['algorithm', 'level', 'entropy']

        entropy_chart = alt.Chart(entropy_data_df).mark_bar().encode(
            x=alt.X('algorithm:N',title=None),
            y=alt.Y('entropy:Q',title=None),
            color='algorithm:N',
            column=alt.Column('level:N', title=None)
        ).properties(width=200)
        st.altair_chart(entropy_chart)
    
    with st.expander(label ="Chi Square Test Statistic of Distributions of Entities per Topic"):
        chisq_data = {alg: get_chisq(alg) for alg in algs}
        chisq_data_df = pd.DataFrame(chisq_data).unstack().reset_index()
        chisq_data_df.columns = ['algorithm', 'level', 'chi_sq']

        chisq_chart = alt.Chart(chisq_data_df).mark_bar().encode(
            x=alt.X('algorithm:N',title=None),
            y=alt.Y('chi_sq:Q',title=None),
            color='algorithm:N',
            column=alt.Column('level:N', title=None)
        ).properties(width=200)
        st.altair_chart(chisq_chart)
    
    st.header("Content Metrics")

    with st.expander(label = "Depth of Pairs of Topics Known to be Related Within Each Discipline"):
        topic_depth = {alg: get_pairwise_depth(alg, "topic") for alg in algs}
        topic_depth_df = pd.DataFrame(topic_depth).unstack().reset_index()      
        topic_depth_df = pd.concat([topic_depth_df, topic_depth_df[0].apply(pd.Series)], axis=1)
        topic_depth_df = topic_depth_df.drop(columns=0).set_index(['level_0', 'level_1']).stack().reset_index()
        topic_depth_df.columns = ['algorithm','domain','discipline','average pairwise depth']

        domain_filter = topic_depth_df["domain"].unique()
        domain_selections = st.multiselect(
            "Choose A Domain, leave blank to view all Domains",
            domain_filter)

        if len(domain_selections)>0:
            topic_depth_df = topic_depth_df.loc[topic_depth_df['domain'].isin(domain_selections)]

        topic_chart = alt.Chart(topic_depth_df).mark_bar().encode(
            x=alt.X('algorithm:N',title=None),
            y=alt.Y('average pairwise depth:Q',title=None),
            color='algorithm:N',
            column=alt.Column('discipline:N', title=None, header=alt.Header(labelFontSize=9))
        ).properties(width=75)
        st.altair_chart(topic_chart)
    
    with st.expander(label = "Depth of Pairs of Subtopics Known to be Related Within Each Topic"):
        subtopic_depth = {alg: get_pairwise_depth(alg, "subtopic") for alg in algs}
        temp = []
        for alg, alg_data in subtopic_depth.items():
            for domain, domain_data in alg_data.items():
                for discipline, discipline_data in domain_data.items():
                    for topic, score in discipline_data.items():
                        temp.append([alg, domain, discipline, topic, score])
        subtopic_df = pd.DataFrame(temp, columns = ['algorithm','domain','discipline','topic','average pairwise depth'])

        domain_filter = subtopic_df["domain"].unique()
        domain_selections = st.multiselect(
            "Choose A Domain, leave blank to view all Domains",
            domain_filter)
        
        discipline_filter = subtopic_df["discipline"].unique()
        discipline_selections = st.multiselect(
            "Choose A Discipline, leave blank to view all Disciplines",
            discipline_filter)

        if len(domain_selections)>0:
            subtopic_df = subtopic_df.loc[subtopic_df['domain'].isin(domain_selections)]
        
        if len(discipline_selections)>0:
            subtopic_df = subtopic_df.loc[subtopic_df['discipline'].isin(discipline_selections)]

        subtopic_chart = alt.Chart(subtopic_df).mark_bar().encode(
            x=alt.X('algorithm:N',title=None),
            y=alt.Y('average pairwise depth:Q',title=None),
            color='algorithm:N',
            column=alt.Column('topic:N', title=None, header=alt.Header(labelFontSize=12))
        ).properties(width=150)
        st.altair_chart(subtopic_chart)



validation_app()
        
