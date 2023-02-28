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

# +
import dap_aria_mapping.getters.openalex as oa
import dap_aria_mapping.getters.taxonomies as tax
import pandas as pd
import altair as alt

# Remove altairs 5000 row limit
alt.data_transformers.disable_max_rows();
# -

test = oa.get_openalex_topics()

test



test = oa.get_openalex_topics(level=5)



len(test_df)

# Calculate the number of topics per document
test_df.loc[:, 'num_topics'] = test_df.loc[:, 'topics'].apply(len)

test_df.head(1)

# Table with histogram of number of topics per document
topics_per_document = test_df.groupby('num_topics').agg(counts=('doc_id', 'count')).reset_index()

# Plot the counts and number of topics per document
(
    alt
    .Chart(topics_per_document)
    .mark_line()
    .encode(
        x=alt.X('num_topics'),
        # log scale for y axis
        y=alt.Y('counts', scale=alt.Scale(type='log')),
    )
)

docs_df = test_df[test_df.num_topics >= 2]

len(docs_df)

works_df = oa.get_openalex_works()

works_df

topic_name_dict = tax.get_topic_names(taxonomy_class='cooccur', name_type='entity', level=5)

# +
# Will need to get
