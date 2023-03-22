# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: dap_aria_mapping
#     language: python
#     name: python3
# ---

# # Testing out charts for novelty analysis
# - Most novel topics (patents or publications, given year)
# - Table with n most and least novel documents per topic?
# - Document counts versus novelty
# - Time series of novelty for topics
# - Growth vs magnitue of novelty

import dap_aria_mapping.utils.novelty_utils as nu
import dap_aria_mapping.getters.novelty as novelty_getters
import dap_aria_mapping.utils.novelty_charts as novelty_charts
import dap_aria_mapping.getters.openalex as oa
import dap_aria_mapping.getters.patents as patents
import dap_aria_mapping.getters.taxonomies as tax
import altair as alt
# remove the altair max row limit
alt.data_transformers.disable_max_rows()
import pandas as pd
pd.set_option('max_colwidth', 500)
import random

# +

import importlib
importlib.reload(novelty_charts);
# -

# # Set up parameters for generating novelty charts

# Taxonomy level to use
LEVEL = novelty_charts.LEVEL

# # Load in data

# OpenAlex paper metadata
openalex_df = oa.get_openalex_works()
# Patent metadata
patents_df = patents.get_patents()

# Get topic-level novelty scores using OpenAlex data
topic_novelty_openalex_df = novelty_getters.get_topic_novelty_openalex(level=LEVEL)
# Get novelty scores of all paper-topic pairs (used to pick out most novel papers in a topic)
openalex_novelty_df = (
    novelty_getters.get_openalex_novelty_scores(level=LEVEL)
    .explode("topics")
    .drop_duplicates(['work_id', 'topics'])
)

# Get topic-level novelty scores using Patent data
topic_novelty_patents_df = novelty_getters.get_topic_novelty_patents(level=LEVEL)
# Get novelty scores of all patent-topic pairs (used to pick out most novel papers in a topic)
patent_novelty_df = (
    novelty_getters.get_patent_novelty_scores(level=LEVEL)
    .explode("topics")
    .drop_duplicates(['publication_number', 'topics'])
)

# +
# topic_pair_novelty_openalex_df = novelty_getters.get_openalex_topic_pair_commonness(level=LEVEL)
# topic_pair_novelty_patents_df = novelty_getters.get_patent_topic_pair_commonness(level=LEVEL)
# -

# Topic names
topic_names = tax.get_topic_names(
    taxonomy_class='cooccur', 
    name_type='entity', 
    level=LEVEL)



# # Charts

# +
# Choose the dataset to use for the charts
dataset = "Publications"

if dataset == "Publications":
    topic_novelty_df = topic_novelty_openalex_df
    document_novelty_df = openalex_novelty_df
    documents_df = openalex_df
    id_column = "work_id"
    title_column = "display_name" 
elif dataset == "Patents":
    topic_novelty_df = topic_novelty_patents_df
    document_novelty_df = patent_novelty_df
    documents_df = patents_df
    id_column = "publication_number"
    title_column = "title_localized"

topic_to_document_dict = nu.create_topic_to_document_dict(document_novelty_df, documents_df, id_column, title_column)
# -

# ## Top topics

most_novel_topics_df = novelty_charts.get_top_novel_topics(
    topic_novelty_df,
    most_novel=True,
)
least_novel_topics_df = novelty_charts.get_top_novel_topics(
    topic_novelty_df,
    most_novel=False,
)

most_novel_topics_df

novelty_charts.chart_top_topics_bubble(most_novel_topics_df)

# ## Topic novelty versus popularity

novelty_charts.chart_topic_novelty_vs_popularity(topic_novelty_df)

# ## Top documents

# ### Top novel documents for a given topic

# +
TOPIC = random.choice(list(topic_to_document_dict.keys()))

print(topic_names[TOPIC])
novelty_charts.get_top_novel_documents(
    document_novelty_df,
    topic_to_document_dict,
    topic_names,
    topic=TOPIC,
    min_topics=5,
)
# -

# ### Top documents containing a given text

# +
EXAMPLE_STRINGS = ['heat pump', 'obesity', 'climate change', 'machine learning']

novelty_charts.check_documents_with_text(
    EXAMPLE_STRINGS[0],
    documents_df,
    document_novelty_df,
    title_column,
    id_column,
    min_n_topics=5,
)

# -

# # OpenAlex vs Patents novelty

topic_novelty_all_data_df = novelty_charts.combine_novelty_datasets(topic_novelty_openalex_df, topic_novelty_patents_df)
topic_novelty_wide_df = novelty_charts.convert_long_to_wide_novelty_table(topic_novelty_all_data_df, topic_names)

importlib.reload(novelty_charts);
novelty_charts.chart_topic_novelty_patent_vs_openalex(topic_novelty_wide_df)


