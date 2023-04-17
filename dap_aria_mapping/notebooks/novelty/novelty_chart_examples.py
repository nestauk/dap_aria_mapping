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
# %%
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

pd.set_option("max_colwidth", 500)
import random

# +

import importlib

importlib.reload(novelty_charts)
# -

# # Set up parameters for generating novelty charts
#
# All parameters are actually set in `novelty_charts` utils file. See the global variables there.

# Taxonomy level to use
LEVEL = novelty_charts.LEVEL

# # Load in data
#
# We'll need the following data:
# - Tables with topic-level novelty scores, based on either OpenAlex or Patent document novelty scores
# - Tables with individual OpenAlex and Patent document novelty scores
# - "Metadata", ie tables with OpenAlex and Patent document data. These are necessary if we want to display the titles of the papers/patents. Hopefully they're not too large, and it will work OK when using the dashboard. Otherwise, I guess you could use polars, or precompute the tables (eg, the tables for "top novel documents")
# - Dictionary with topic names. You will probably need to change this to use the ChatGPT names.

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
    .drop_duplicates(["work_id", "topics"])
)

# Get topic-level novelty scores using Patent data
topic_novelty_patents_df = novelty_getters.get_topic_novelty_patents(level=LEVEL)
# Get novelty scores of all patent-topic pairs (used to pick out most novel papers in a topic)
patent_novelty_df = (
    novelty_getters.get_patent_novelty_scores(level=LEVEL)
    .explode("topics")
    .drop_duplicates(["publication_number", "topics"])
)

# Topic names
topic_names = tax.get_topic_names(
    taxonomy_class="cooccur", name_type="chatgpt", n_top=35, level=LEVEL, postproc=True
)
topic_names = {k: v["name"] for k, v in topic_names.items()}
# %%
# # Charts

# ## Choose dataset to visualise
#
# I haven't tried to combine the novelty scores from OpenAlex or patents into one combined score. Therefore, the user will have to choose between the two datasets when inspecting most of the charts.
#
# In my very short exploration, it feels like it's easier to understand the OpenAlex paper titles, so that might be a better default choice.

# +
# Choose the dataset to use for the charts
dataset = "Publications"
# dataset = "Patents"

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

# %%
topic_to_document_dict = nu.create_topic_to_document_dict(
    document_novelty_df, documents_df, id_column, title_column
)
# -
# %%
# ## Top documents containing a given text
#
# **Note to developers:**
#
# I feel that perhaps the most intuitive way to introduce the users to the novelty score, is to show examples of very specific selection of documents - and highlighting the most and least novel documents according to our score.
#
# In this way, the domain of the papers is very clear, and user can judge the way the novetly score is working.
#
# Therefore, I would propose starting with selecting documents based on easily understandable keywords, for example "heat pump" or "obesity".
#
# If this is too tricky to implement in the dashboard, perhaps you could precompute the tables for a few specific examples.

# **Method description:**
#
# Our novelty score indicates how uncommon are the combinations of topics in a document. The approach is adapted from a paper by [Lee, Walsh and Wang (2015)](https://www.sciencedirect.com/science/article/abs/pii/S0048733314001826), where the authors measured novelty by considering the combinations of paper citations (this measure [has been shown to correlate](https://www.sciencedirect.com/science/article/abs/pii/S1751157718304371) with what researchers consider novel papers). Here, instead of citations we are using the topics detected in the paper or patent abstracts.
#
# Briefly, the score is derived by first calculating the "commonness" of each pairwise combination of topics, across all documents in a given year:
# ```
# commonness(i,j) = N(i,j,t) * N(t) / (N(i,t)*N(j,t))
#
# where:
# - N(i,j,t) is the number of co-occurrences of i and j in time t (t = usually in a given year)
# - N(i, t) is the number of pairs of topics that include topic i
# - N(j, t) is the number of pairs of topics that include topic j
# - N(t) is the number of pairs of topics in time t
# ```
#
# Then, the novelty score of each document if calculated by taking the negative natural logarithm of the 10th percentile of its distribution of topic pair commonness scores.
#
# Note that in this way, we're measuring novelty with respect to other papers published in the same year (as in the original paper by Lee et al.). Further modifications could be used to facilitate comparisons of papers across years: eg, by altering the time window for calculating commonness to include previous years, or by normalising the novelty scores in each year by z-scoring.

# **Table: Document novelty scores**
#
# In the first example, we demonstrate the most and least novel papers containing the
# phrase "heat pump" (a low-carbon heating technology).
#
# We see that the papers with the lowest novelty score appear more generic, ie concerned with heating or efficiency, whereas papers with the highest novelty score appear more interdisciplinary, eg involving computer simulations or less common applications of the technology.
#
# You can explore other search phrases, to develop intuition for the novelty score.

# +
EXAMPLE_STRINGS = ["heat pump", "obesity", "climate change", "machine learning"]

search_phrase = EXAMPLE_STRINGS[0]
# %%
novelty_charts.check_documents_with_text(
    search_phrase,
    documents_df,
    document_novelty_df,
    title_column,
    id_column,
    min_n_topics=5,
).head(5)

# -

# Most novel
novelty_charts.check_documents_with_text(
    search_phrase,
    documents_df,
    document_novelty_df,
    title_column,
    id_column,
    min_n_topics=5,
).tail(5)
# %%
# ## Top topics
#
# **Table/Chart: Most and least novel topics**
#
# We have characterised the overall novelty of a given topic, by aggregating (ie, taking the median) the novelty scores of all documents that featured this topic in their abstracts.
#
# In this way, a high topic-level novelty score indicates that the topic is associated with documents with high novelty scores.
#
# Note that in now way do we imply that topics with low novelty scores are not performing innovative research and improving the scientific knowlege. Rather, the novelty score indicates that the topic is associated with documents with more common topic combinations. This might be the case, for example, for topics associated with less interdisciplinary research - in those cases, the research can still be innovative, but it might be more confined within a single field of research.

topic_novelty_df
topic_novelty_df["topic_name"] = topic_novelty_df["topic_name"].apply(
    lambda x: x["name"]
)

most_novel_topics_df = novelty_charts.get_top_novel_topics(
    topic_novelty_df,
    most_novel=True,
)
least_novel_topics_df = novelty_charts.get_top_novel_topics(
    topic_novelty_df,
    most_novel=False,
)

# %%
novelty_charts.chart_top_topics_bubble(most_novel_topics_df)
# %%
# ## Topic novelty versus popularity
#
# We can inspect the topic novelty scores versus their popularity (ie, number of documents in which they appear). This shows that, while there is a tendency for very popular topics to have higher novelty scores, there is still a fairly large spread of novelty scores for topics with similar, intermediate-level of popularity.

novelty_charts.chart_topic_novelty_vs_popularity(topic_novelty_df)
# %%
# ## Top novel documents for a given topic
#
# Once you have identified a topic of interest, you can also inspect the most or least novel documents for that particular topic.

importlib.reload(novelty_charts)

# +
TOPIC = random.choice(list(topic_to_document_dict.keys()))

print(topic_names[TOPIC])
novelty_charts.get_top_novel_documents(
    document_novelty_df,
    topic_to_document_dict,
    topic_names,
    topic=TOPIC,
    min_topics=5,
    id_column=id_column,
    title_column=title_column,
)
# %%
# -

# # Time series
#
# It might also be interesting to look at the temporal evolution of topic novelty scores. Some topics exhibit rapid changes in their involvement in novel research, which could be a signal of interesting developments in the field.
#
#

# +
TOPICS_FOR_TIME_SERIES = novelty_charts.get_top_novel_topics(
    topic_novelty_df
).topic.to_list()

novelty_charts.chart_topic_novelty_timeseries(
    topic_novelty_df,
    TOPICS_FOR_TIME_SERIES,
    show_legend=False,
)
# %%
# -

# ## Magnitude and growth of novelty
#
# To summarise the growth trends across all topics, we can visualise the growth of the novelty scores for each topic across the most recent five-year time period - and compare it with the topics' average novelty score in the same time period.
#
# In this way, we can distingush four types of trends:
# - Emerging topics: Low average novelty, but high recent growth
# - Hot topics: High average novelty and high recent growth
# - Stable uncommon topics: High average novelty and low recent growth
# - Stable common topics: Low average novelty and low recent growth
#
# To distinguish between low and high average novelty, we can simply use the median average novelty score across all topics (indicated in the charts with a vertical line).
#
# It might be of particular interest to investigate the emerging and hot topics, as the high growth might signal interesting developments in the field.
#
# Note that we use the phrase "uncommon topic" here to imply that the topic is involved in uncommon types of research (whereas the topic itself might be very popular).

magnitude_growth_df = novelty_charts.calculate_magnitude_growth(
    topic_novelty_df, topic_names
)

# Topics with the highest growth in novelty
novelty_charts.get_topics_with_highest_growth(magnitude_growth_df)
# %%
# **Note to developers:**
#
# I've split the growth versus magnitude charts shown below in two parts (positive and negative growth). This is because it's better to use log vertical axis as the growth rates vary a lot. However, I'm not sure if it's possible in altair to show both positive and negative growth rates AND use logarithmic axis in the same chart. Therefore, for the final dashboard, you can probably simply concatenate these two charts or show them directly one after the other (NB: in the latter case, you might want to set x-axis limits to be the same for both charts).
#
# I did try to concatenate them using `vconcat` but got errors (don't know why).

# Topics with positive growth
novelty_charts.chart_magnitude_growth(
    magnitude_growth_df, growth=True, show_doc_counts=True
)
# %%
# Topics with negative growth
novelty_charts.chart_magnitude_growth(
    magnitude_growth_df, growth=False, show_doc_counts=True
)
# %%
# # OpenAlex vs Patents novelty
#
# Finally, it is interesting to compare the topic-level novelty scores of the research publication and patents datasets. Here, we have normalised the novelty scores in each dataset by z-scoring them.
#
# In this way, we can identify topics that have higher novelty scores for one of the datasets, potentially signalling different approaches to research and development in the academic (OpenAlex) and industrial (Patents) settings.

topic_novelty_all_data_df = novelty_charts.combine_novelty_datasets(
    topic_novelty_openalex_df, topic_novelty_patents_df
)
topic_novelty_wide_df = novelty_charts.convert_long_to_wide_novelty_table(
    topic_novelty_all_data_df, topic_names
)

novelty_charts.chart_topic_novelty_patent_vs_openalex(topic_novelty_wide_df)
# %%
