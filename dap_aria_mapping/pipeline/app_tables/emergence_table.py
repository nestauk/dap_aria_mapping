from dap_aria_mapping.getters.openalex import get_openalex_topics, get_openalex_works
from dap_aria_mapping.getters.patents import get_patent_topics, get_patents
import polars as pl
import pandas as pd
from collections import defaultdict

if __name__ == "__main__":
    #swap this once we decide what taxonomy we're using
    patents_with_topics = get_patent_topics(tax = "cooccur", level = 3)

    patent_dates = pl.DataFrame(get_patents()).select(["publication_number", "publication_date"]).with_columns(pl.col("publication_date").dt.year().alias("year"))

    temp = defaultdict(lambda: defaultdict(int))

    for doc, doc_topics in patents_with_topics.items():
        year = patent_dates.filter(
        pl.col("publication_number") == doc
    ).select("year").item()
        for topic in doc_topics:
            temp[year][topic]+=1
    
    patents_df = pl.DataFrame(pd.DataFrame.from_dict(temp,orient='index').unstack().reset_index())
    patents_df.columns = ['topic', 'year', 'patent_count']
    patents_df.with_columns(
        (pl.col("topic").str.split("_").arr.get(0)).alias("domain"),
        (pl.col("topic").str.split("_").arr.get(1)).alias("area"))

    
    