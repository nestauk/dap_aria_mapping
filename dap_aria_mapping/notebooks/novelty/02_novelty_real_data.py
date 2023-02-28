# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: dap_aria_mapping
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Measure novelty of papers

# %%
import dap_aria_mapping.utils.novelty_utils as nu
import importlib
importlib.reload(nu);

# %%
import dap_aria_mapping.getters.openalex as oa
import dap_aria_mapping.getters.taxonomies as tax

# %%
import pandas as pd

# %% [markdown]
# ## Load and process data

# %%
# Metadata
works_df = oa.get_openalex_works()

# %%
# Load topics for all research papers
topics_dict = oa.get_openalex_topics()

# %%
topics_df = (
    # Convert to dataframe
    pd.DataFrame(
        data={
            'work_id': list(topics_dict.keys()),
            'topics': list(topics_dict.values()),
        }
    )
    # Deduplicate the lists in 'topics' column in each row
    .assign(topics=lambda df: df['topics'].apply(lambda x: list(set(x))))
    # Drop rows with less than two topics
    .assign(no_topics=lambda df: df['topics'].apply(lambda x: len(x)))
    .query('no_topics >= 2')
    # Merge with metadata to get publication year
    .merge(works_df[['work_id', 'publication_year']], on='work_id', how='left')
    .rename(columns={'publication_year': 'year'})
    # Drop rows with missing year
    .dropna(subset=['year'])
    .astype({'year': 'int'})
)

# %%
test_df = topics_df.sample(1000)
test_df

# %% [markdown]
# ## Calculate commonness

# %%
importlib.reload(nu);

# %%
work_novelty_df = (
    nu.document_novelty(topics_df, id_column='work_id')
    .merge(test_df[['work_id', 'topics']], on='work_id', how='left')
    .assign(no_topics=lambda df: df['topics'].apply(lambda x: len(x)))
)

# %%
work_novelty_df

# %%
