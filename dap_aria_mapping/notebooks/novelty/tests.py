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

import dap_aria_mapping.getters.openalex as oa
import dap_aria_mapping.getters.taxonomies as tax
import pandas as pd

test = oa.get_openalex_topics()

test



test = oa.get_openalex_topics(level=5)

test_df = pd.DataFrame(
    data={
        'doc_id': list(test.keys()),
        'topics': list(test.values()),
    }
)

test_df.sample(50)

works_df = oa.get_openalex_works()

works_df

topic_name_dict = tax.get_topic_names(taxonomy_class='cooccur', name_type='entity', level=5)

# +
# Will need to get
