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

# # Adapting novelty score U
#
# The first novelty indicator denoted as “novelty score U” has been proposed by Uzzi et al. (2013) and revised by Lee et al. (2015). 
#
# The novelty score U is calculated as follows: 
#
# commonness(i,j) = N(i,j,t) * N(t) / (N(i,t)*N(j,t))
#
# where:
# - N(i,j,t) is the number of co-occurrences of i and j in time t, 
# - N(i, t) is the number of pairs of topics that include topic i
# - N(j,t) is the number of pairs of topics that include topic j
# - N(t) is the number of pairs of topics in time t
#

import pandas as pd
import numpy as np
from typing import Tuple
pd.set_option('display.max_colwidth', None)
import itertools


# +
# Generate a dummy table that has columns "topic_pair", "year" and "counts"
def generate_random_topic_pair(n_topics: int=100) -> Tuple[str, str]:
    """
    Generate two random integers between 0 and n_topics-1, make sure they are different
    Order them such that the smaller one is first
    Then convert them to strings of the form 'topic_x' where x is the integer        

    Args:
        n_topics (int, optional): Number of topics to generate. Defaults to 100.

    Returns:
        tuple(str, str): A tuple of two strings of the form 'topic_x' where x is an integer
    """
    while True:
        topic1 = np.random.randint(0, n_topics)
        topic2 = np.random.randint(0, n_topics)
        if topic1 != topic2:
            break
    if topic1 > topic2:
        topic1, topic2 = topic2, topic1
    topic1 = f"topic_{topic1}"
    topic2 = f"topic_{topic2}"
    return topic1, topic2
    
def generate_dummy_table(n_topics: int = 20, n_rows: int = 1000):    
    """
    Generate a dummy table that has columns "topic_pair", "year" and "counts"
    Topic pairs are strings of the form 'topic_x' where x is an integer between 0 and n_topics-1

    Args:
        n_topics (int, optional): Number of topics to generate. Defaults to 100.
        n_rows (int, optional): Number of rows to generate. Defaults to 1000.
        
    Returns:
        pd.DataFrame: A dummy table
    """

    # Generate topic pairs and splint into two arrays
    topic_pairs = np.array([generate_random_topic_pair(n_topics) for _ in range(n_rows)])
    topic1 = topic_pairs[:, 0]
    topic2 = topic_pairs[:, 1]
    # Year and counts are random integers
    years = np.random.choice(range(2000, 2020), size=(n_rows, 1)).squeeze()
    counts = np.random.choice(range(100), size=(n_rows, 1)).squeeze()
    # Create a dataframe
    return (
        pd.DataFrame(data={
            'topic_1': topic1,
            'topic_2': topic2,
            'year': years,
            'counts': counts
        })
        .groupby(['topic_1', 'topic_2', 'year'], as_index=False)
        .agg({'counts': 'sum'})
    )



# +
# Plot a random chart showing the evolution of counts for a random topic pair
def select_random_existing_topic_pair(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Select a random topic pair from the existing topic pairs

    Args:
        df (pd.DataFrame): Dataframe with columns "topic_1", "topic_2", "year" and "counts"

    Returns:
        Tuple[str, str]: A tuple of two strings of the form 'topic_x' where x is an integer
    """
    # Select a random topic pair from the existing topic pairs
    topic_pairs = df[['topic_1', 'topic_2']].drop_duplicates().values
    topic1, topic2 = topic_pairs[np.random.randint(0, len(topic_pairs))]
    return topic1, topic2

def plot_random_topic_pair(df: pd.DataFrame, topic1: str = None, topic2: str = None):
    """
    Plot a random chart showing the evolution of counts for a random topic pair

    Args:
        df (pd.DataFrame): Dataframe with columns "topic_1", "topic_2", "year" and "counts"
    """
    # If no topics given select random
    if topic1 is None or topic2 is None:
        topic1, topic2 = select_random_existing_topic_pair(df)
    # Filter the dataframe
    df = df[(df['topic_1'] == topic1) & (df['topic_2'] == topic2)]
    # Imput empty years with 0 counts
    df = df.set_index('year').reindex(range(2000, 2020)).fillna(0).reset_index()
    # Plot
    print(f"Plotting {topic1} + {topic2}")
    df.plot(x='year', y='counts')


# -

dummy_df = generate_dummy_table(n_rows=10000)
dummy_df.head(3)


# Calculate the commonness measure
def com(df: pd.DataFrame, year: int, topic_i: str, topic_j: str) -> float:
    """
    Calculate the commonness measure for a given topic pair and year

    commonness(i,j) = N(i,j,t) * N(t) / (N(i,t)*N(j,t))

    where:
    - t is the year
    - N(i,j,t) is the number of co-occurrences of i and j in year t, 
    - N(i, t) is the number of pairs of topics that include topic i in year t
    - N(j,t) is the number of pairs of topics that include topic j in year t
    - N(t) is the number of pairs of topics in year t    

    Args:
        df (pd.DataFrame): Dataframe with columns "topic_1", "topic_2", "year" and "counts"
        year (int): Year to calculate the novelty measure for
        topic1 (str): First topic of the pair
        topic2 (str): Second topic of the pair

    Returns:
        float: The commonness measure
    """
    # Calculate N(i,j,t)
    N_ij_t = df[(df['topic_1'] == topic_i) & (df['topic_2'] == topic_j) & (df['year'] == year)]['counts'].sum()
    # Calculate N(i,t)
    N_i_t = df[((df['topic_1'] == topic_i) | (df['topic_2'] == topic_i)) & (df['year'] == year)]['counts'].sum()
    # Calculate N(j,t)
    N_j_t = df[((df['topic_1'] == topic_j) | (df['topic_2'] == topic_j)) & (df['year'] == year)]['counts'].sum()
    # Calculate N(t)
    N_t = df[df['year'] == year]['counts'].sum()
    # Calculate commonness
    comm =  N_ij_t * N_t / (N_i_t * N_j_t)    
    # Return all calculated variables
    return comm, N_ij_t, N_i_t, N_j_t, N_t



# +
topic1, topic2 = select_random_existing_topic_pair(dummy_df)

com(dummy_df, year=2010, topic_i=topic1, topic_j=topic2)


# +
# Generate a dummy table with document ids and list of topics mentioned in the document
def generate_dummy_document_table(n_documents: int=1000, n_topics: int=20, max_n_topics_per_document: int=5) -> pd.DataFrame:
    """
    Generate a dummy table that has columns "document_id", "years" and "topics".

    Args:
        n_documents (int, optional): Number of documents to generate. Defaults to 1000.
        n_topics (int, optional): Number of unique topics to generate. Defaults to 20.
        max_n_topics_per_document (int, optional): Maximum number of topics per document. Defaults to 5.

    Returns:
        pd.DataFrame: A dataframe with columns "document_id", "years" and "topics"
    """
    # Generate document ids
    document_ids = [f"document_{i}" for i in range(n_documents)]
    # Generate a list of random topics for each document like this [topic_1, topic_2, ...]
    # Make sure that all topics are unique for each document
    topics = [
        [f"topic_{i}" for i in np.random.choice(range(n_topics), size=np.random.randint(1, max_n_topics_per_document), replace=False)]
        for _ in range(n_documents)
    ]
    # Years
    years = np.random.choice(range(2000, 2020), size=(n_documents, 1)).squeeze()
    # Create a dataframe
    return pd.DataFrame(data={'document_id': document_ids, 'topics': topics, 'year': years})

# Create a document - topic pair table
def find_document_topic_pairs(document_table: pd.DataFrame) -> pd.DataFrame:
    """
    Given a document table with document ids, years and list of topics, create a table with document_ids
    and all pairwise combinations of topics in respective documents

    Args:
        document_table (pd.DataFrame): Dataframe with columns "document_id", "years" and "topics"

    Returns:
        pd.DataFrame: A dataframe with columns "document_id", "topic_1", "topic_2", "year"
    """
    # Create a list of all document - topic pairs
    document_topic_pairs = []
    for i, row in document_table.iterrows():
        document_id = row['document_id']
        year = row['year']
        topics = row['topics']
        for topic1, topic2 in itertools.combinations(topics, 2):
            document_topic_pairs.append([document_id, topic1, topic2, year])
    # Create a dataframe
    return pd.DataFrame(data=document_topic_pairs, columns=['document_id', 'topic_1', 'topic_2', 'year'])


# -

doc_df = generate_dummy_document_table()
doc_topics_df = find_document_topic_pairs(doc_df)
doc_topics_df


# +
# Given a table with document ids, year and topic pairs mentioned in the document,
# calculate the commonness measure for each topic pair and year
def calculate_commonness_for_document_topic_pairs(document_topic_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Given a table with document ids, year and topic pairs mentioned in the document,
    calculate the commonness measure for each topic pair and year

    Args:
        document_topic_pairs (pd.DataFrame): Dataframe with columns "document_id", "topic_1", "topic_2", "year"

    Returns:
        pd.DataFrame: A dataframe with columns "topic_1", "topic_2", "year", "commonness"
    """
    # Calculate the number of co-occurrences of each topic pair in each year
    df = document_topic_pairs.groupby(['topic_1', 'topic_2', 'year']).size().reset_index(name='counts')
    # Calculate the commonness measure for each topic pair and year
    df['commonness'] = df.apply(lambda x: com(df, x['year'], x['topic_1'], x['topic_2'])[0], axis=1)
    # Return the dataframe
    return df

def calculate_commoness_for_document_topics(document_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a document table with document ids, years and list of topics, calculate the commonness measure
    for each topic pair and year

    Args:
        document_df (pd.DataFrame): Dataframe with columns "document_id", "years" and "topics"

    Returns:
        pd.DataFrame: A dataframe with columns "topic_1", "topic_2", "year", "commonness"
    """
    # Create a table with document ids, year and topic pairs mentioned in the document
    document_topic_pairs = find_document_topic_pairs(document_df)
    # Calculate the commonness measure for each topic pair and year
    return document_topic_pairs.merge(
        calculate_commonness_for_document_topic_pairs(document_topic_pairs),
        on=['topic_1', 'topic_2', 'year'],
        how='left',
    )


# -

calculate_commoness_for_document_topics(doc_df)


# Calculate commonness of a document by aggregating the commonness of all topic pairs mentioned in the document
# Provide option to choose different aggregation method
def calculate_commonness_for_document(document_df: pd.DataFrame, aggregation_method: str='mean') -> pd.DataFrame:
    """
    Calculate commonness of a document by aggregating the commonness of all topic pairs mentioned in the document

    Args:
        document_df (pd.DataFrame): Dataframe with columns "document_id", "years" and "topics"
        aggregation_method (str, optional): Aggregation method to use. Defaults to 'mean'.

    Returns:
        pd.DataFrame: A dataframe with columns "document_id", "year", "commonness"
    """
    # Calculate the commonness measure for each topic pair and year
    df = calculate_commoness_for_document_topics(document_df)
    # Aggregate the commonness measure for each document and year
    df = df.groupby(['document_id', 'year']).agg({'commonness': aggregation_method}).reset_index()
    # Return the dataframe
    return df


calculate_commonness_for_document(doc_df)




