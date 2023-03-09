import polars as pl

def count_documents(doc_df: pl.DataFrame) -> pl.DataFrame:
    """takes a dataframe with documents (including their topic and year) 
    and generates a count of documents per topic per year

    Args:
        doc_df (pl.DataFrame): documents with topics and years

    Returns:
        pl.DataFrame: count of documents per topic per year
    """
    q = (doc_df.lazy().groupby(["topic", "year"]).agg([pl.count()]))
    return q.collect()


def expand_topic_col(topic_df: pl.DataFrame) -> pl.DataFrame:
    """takes a dataframe with a column called "topic" (level 3 of taxonomy), 
        and splits the column to add area ("level 2") and domain ("level 1")

    Args:
        topic_df (pl.DataFrame): dataframe with column for a "topic"

    Returns:
        pl.DataFrame: the same dataframe with "area" and "domain" columns added
    """
    return topic_df.with_columns(
        (pl.col("topic").str.split("_").arr.get(0)).alias("domain"),
        (pl.col("topic").str.split("_").arr.get(0) + "_" + pl.col("topic").str.split("_").arr.get(1)).alias("area"))