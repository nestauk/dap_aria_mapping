"""
Utils for generating novelty measurement charts
"""
import pandas as pd
from typing import List, Dict
import altair as alt

# remove the altair max row limit
alt.data_transformers.disable_max_rows()
import itertools

alt.data_transformers.disable_max_rows()

# Which novelty measure to use
NOVELTY_MEASURE = "topic_doc_novelty"
# Taxonomy level to use
LEVEL = 3
# Year for which to display the results
YEAR = 2021
# Number of top topics to show
TOP_N_TOPICS = 25
# Number of documents to show
TOP_N_DOCS = 10

# Columns to show for top topics
TOP_TOPIC_COLUMNS = ["topic", "topic_name", "novelty_score", "doc_counts"]


def get_top_novel_topics(
    topic_novelty_df: pd.DataFrame,
    most_novel: bool = True,
    novelty_measure: str = NOVELTY_MEASURE,
    year: int = YEAR,
    top_n: int = TOP_N_TOPICS,
    columns_to_show=TOP_TOPIC_COLUMNS,
) -> pd.DataFrame:
    """
    Get the most or least "novel" topics for a given year

    Args:
        topic_novelty_df (pd.DataFrame): Topic novelty dataframe
        most_novel (bool, optional): Whether to get the most novel topics or
            the least novel topics.
        novelty_measure (str, optional): Novelty measure to use.
        year (int, optional): Year for which to get the most/lesat novel topics.
        top_n (int, optional): Number of topics to return.
        columns_to_show (list, optional): Columns to show in the output.

    Returns:
        pd.DataFrame: Table with the most/least novel topics for the given year
    """
    df = topic_novelty_df.query("year == @year").sort_values(
        novelty_measure, ascending=False
    )
    if most_novel:
        df = df.head(top_n)
    else:
        df = df.tail(top_n)
    return (
        df.reset_index(drop=True).rename(columns={NOVELTY_MEASURE: "novelty_score"})
    )[columns_to_show]


def get_top_novel_documents(
    document_novelty_df: pd.DataFrame,
    topic_to_document_dict: Dict[str, List[str]],
    topic_names: Dict[str, str],
    most_novel: bool = True,
    topic: str = None,
    year: int = YEAR,
    top_n: int = TOP_N_DOCS,
    min_topics: int = 2,
    id_column = "work_id",
    title_column = "display_name",
    output_columns: list = [
        "novelty",
        "n_topics",
        "topic",
        "topic_name",
    ],
) -> pd.DataFrame:
    """
    Get the most or least "novel" documents for a given year

    Args:
        document_novelty_df (pd.DataFrame): Document novelty dataframe
        topic_to_document_dict (Dict[str: List[str]]): Dictionary mapping topics to documents
        topic_names (Dict[str: str]): Dictionary mapping topic labels to topic names
        most_novel (bool, optional): Whether to get the most novel documents or the least novel documents.
        topic (str, optional): Topic for which to get the mostl/least novel documents. If None, get the most novel documents for all topics.
        year (int, optional): Year for which to get the most/lesat novel documents.
        top_n (int, optional): Number of documents to return.
        min_topics (int, optional): Minimum number of topics a document
            must be associated with to be included in the results.
        output_columns (list, optional): Columns to show in the output.

    Return:
        pd.DataFrame: Table with the most/least novel documents for the given year.
    """
    output_columns = [id_column, title_column] + output_columns
    if topic is not None:
        # get all documents for the specified topic
        _df = pd.DataFrame(topic_to_document_dict[topic])
    else:
        # get all documents for all topics
        _df = document_novelty_df.drop_duplicates(id_column)
    # select documents for the given year
    _df = (
        _df.query("year == @year")
        .query("n_topics >= @min_topics")
        .sort_values("novelty", ascending=False)
    )
    # most or least novel
    if most_novel:
        _df = _df.head(top_n)
    else:
        _df = _df.tail(top_n)
    # add topic names and return
    return (
        _df.reset_index(drop=True)
        .assign(topic=topic)
        .assign(topic_name=lambda df: df["topic"].map(topic_names))
    )[output_columns]


def check_documents_with_text(
    text: str,
    documents_df: pd.DataFrame,
    document_novelty_df: pd.DataFrame,
    title_column: str = "display_name",
    id_column: str = "work_id",
    min_n_topics: int = 5,
) -> pd.DataFrame:
    """
    Find most and least novel documents with given text in their title

    Args:
        text (str): Text to search for in document titles
        documents_df (pd.DataFrame): Documents dataframe
        document_novelty_df (pd.DataFrame): Document novelty dataframe
        title_column (str, optional): Column in documents_df containing document titles
        id_column (str, optional): Column in documents_df containing document IDs
        min_n_topics (int, optional): Minimum number of topics a document
            must be associated with to be included in the results.

    Returns:
        pd.DataFrame: Table with the all documents containg the text in the title, sorted by novelty
    """
    bool_vector = documents_df[title_column].str.lower().str.contains(text).astype(bool)
    return (
        documents_df.loc[bool_vector, :]
        .merge(
            document_novelty_df[
                [id_column, "year", "novelty", "n_topics"]
            ].drop_duplicates(id_column),
            on=id_column,
            how="left",
        )[[title_column, id_column, "year", "novelty", "n_topics"]]
        .dropna()
        .query("n_topics >= @min_n_topics")
        .sort_values("novelty", ascending=True)
    )


def combine_novelty_datasets(
    topic_novelty_openalex_df: pd.DataFrame, topic_novelty_patents_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine novelty datasets for OpenAlex and patents

    Args:
        topic_novelty_openalex_df (pd.DataFrame): Dataframe with topic novelty scores for OpenAlex
        topic_novelty_patents_df (pd.DataFrame): Dataframe with topic novelty scores for patents

    Returns:
        pd.DataFrame: Dataframe with topic novelty scores for OpenAlex and patents
    """
    return pd.concat(
        [
            topic_novelty_openalex_df.assign(source="OpenAlex"),
            topic_novelty_patents_df.assign(source="Patents"),
        ],
        ignore_index=True,
    )


def convert_long_to_wide_novelty_table(
    topic_novelty_df: pd.DataFrame,
    topic_names: Dict[str, str],
    year: int = YEAR,
) -> pd.DataFrame:
    """
    Convert long novelty table to wide novelty table, and normalise novelty scores for
    OpenAlex publications and patents using z-scores

    Args:
        topic_novelty_df (pd.DataFrame): Long novelty table with columns for
            topic, year, novelty_score, source
        topic_names (Dict[str, str]): Dictionary mapping topic labels to topic names
        year (int, optional): Year for which to use novelty scores

    Returns:
        pd.DataFrame: Wide novelty table with columns:
            - topic: topic label
            - topic_name: topic name
            - OpenAlex: novelty score for OpenAlex
            - Patents: novelty score for patents
            - openalex_zscore: z-score of novelty score for OpenAlex
            - patents_zscore: z-score of novelty score for patents
    """
    return (
        topic_novelty_df.query("year == @year")
        .pivot(index="topic", columns="source", values=NOVELTY_MEASURE)
        .reset_index()
        # topic names
        .assign(topic_name=lambda df: df["topic"].map(topic_names))
        # normalise OpenAlex and Patents novelty scores using z-scores
        .assign(
            openalex_zscore=lambda df: (df["OpenAlex"] - df["OpenAlex"].mean())
            / df["OpenAlex"].std(),
            patents_zscore=lambda df: (df["Patents"] - df["Patents"].mean())
            / df["Patents"].std(),
        )
    )


def smoothed_growth_rate(
    df: pd.DataFrame,
    variable: str = NOVELTY_MEASURE,
    start_year: int = 2018,
    end_year: int = 2022,
) -> float:
    """
    Uses a three-year rolling average to calculate the relative growth

    Args:
        df (pd.DataFrame): Dataframe with columns for year and novelty score for a given topic
        variable (str, optional): Column in df containing novelty score
        start_year (int, optional): Start year for calculating growth rate
        end_year (int, optional): End year for calculating growth rate

    Returns:
        float: Smoothed estimate of the relative growth rate of novelty
    """
    start_years = list(range(start_year - 2, start_year))
    end_years = list(range(end_year - 2, end_year))
    start_novelty = df.query("year in @start_years")[variable].mean()
    end_novelty = df.query("year in @end_years")[variable].mean()
    return (end_novelty - start_novelty) / start_novelty


def calculate_magnitude_growth(
    topic_novelty_df: pd.DataFrame,
    topic_names: Dict[str, str],
    start_year: int = 2018,
    end_year: int = 2022,
) -> pd.DataFrame:
    """Calculate magnitude and growth of novelty for each topic. The magnitude is the average
    novelty score for the topic, and the growth is the smoothed estimate of the relative growth
    rate of novelty.

    Args:
        topic_novelty_df (pd.DataFrame): Dataframe with topic novelty scores
        topic_names (Dict[str, str]): Dictionary mapping topic labels to topic names
        start_year (int, optional): Start year for calculating growth rate
        end_year (int, optional): End year for calculating growth rate

    Returns:
        pd.DataFrame: Dataframe with columns:
            - topic: topic label
            - topic_name: topic name
            - avg_novelty: average novelty score for topic
            - growth: smoothed estimate of the relative growth rate of novelty
            - doc_counts: total number of documents for topic in the specified time period
    """
    unique_topics = topic_novelty_df.topic.unique()
    years = list(range(start_year, end_year + 1))

    avg_novelty = []
    growth = []
    doc_counts = []
    for topic in unique_topics:
        # get novelty scores for specific topic
        topic_df = topic_novelty_df[topic_novelty_df.topic == topic]
        # calculate average novelty score for topic
        avg_novelty.append(topic_df.query("year in @years")[NOVELTY_MEASURE].mean())
        # calculate growth rate for topic
        growth.append(
            smoothed_growth_rate(topic_df, start_year=start_year, end_year=end_year)
        )
        doc_counts.append(topic_df.query("year in @years").doc_counts.sum())

    return pd.DataFrame(
        {
            "topic": unique_topics,
            "avg_novelty": avg_novelty,
            "growth": growth,
            "doc_counts": doc_counts,
        }
    ).assign(
        topic_name=lambda x: x.topic.map(topic_names),
    )


def get_topics_with_highest_growth(
    magnitude_growth_df: pd.DataFrame,
    growth: bool = True,
    top_n: int = TOP_N_TOPICS,
) -> pd.DataFrame:
    """
    Get topics with highest growth or decline in novelty

    Args:
        magnitude_growth_df (pd.DataFrame): Dataframe with columns for topic, topic_name,
            avg_novelty, growth
        growth (bool, optional): Whether to get topics with highest growth or decline. Defaults to True.
        top_n (int, optional): Number of topics to return.

    Returns:
        pd.DataFrame: Dataframe with columns for topic, topic_name, avg_novelty, growth
    """
    if growth:
        return magnitude_growth_df.nlargest(top_n, "growth")
    else:
        return magnitude_growth_df.nsmallest(top_n, "growth")


def chart_top_topics_bubble(
        data: pd.DataFrame, values_label: str = "Topic novelty"
) -> alt.Chart:
    """
    Chart top topics as a bubble chart

    Args:
        data (pd.DataFrame): Dataframe with top topics with the
            following columns: topic_name, novelty_score, doc_counts

    Returns:
        alt.Chart: A bubble chart with the top topics on the x-axis
            and the number of documents in the topic on the y-axis
    """
    # A bubble chart
    bubbles = (
        alt.Chart(data)
        .mark_circle()
        .encode(
            x=alt.X(
                "novelty_score",
                title=values_label,
                # make the x-axis not start from 0
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "topic_name",
                title="",
                sort=alt.EncodingSortField(field="novelty_score", order="descending"),
                # hide the y-axis
                axis=alt.Axis(labels=False, ticks=False, domain=False),
            ),
            size=alt.Size(
                "doc_counts",
                title="Number of documents",
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("topic_name", title="Topic title"),
                alt.Tooltip("novelty_score", title="Topic novelty (document-based)"),
                alt.Tooltip("doc_counts", title="Number of documents"),
            ],
        )
    )

    # Add text labels (don't scale with bubble size)
    labels = bubbles.mark_text(align="left", baseline="middle", dx=7).encode(
        # limit text length
        text=alt.Text("topic_name:N"),
        size=alt.Size(),
    )

    # Combine the two charts
    return bubbles + labels  # .properties(width=600, height=600)


def chart_topic_novelty_vs_popularity(
    topic_novelty_df: pd.DataFrame,
    novelty_measure: str = NOVELTY_MEASURE,
    novelty_measure_label: str = "Topic novelty",
    year: int = YEAR,
    min_docs: int = 50,
) -> alt.Chart:
    """
    Chart topic novelty vs. number of documents

    Args:
        topic_novelty_df (pd.DataFrame): Topic novelty dataframe with columns
            topic_name, topic_doc_novelty, doc_counts
        year (int, optional): Year for which to get the novelty values
        min_docs (int, optional): Minimum number of documents a topic must have to be included in the chart

    Returns:
        alt.Chart: A scatter plot with topic novelty on the x-axis and
            number of documents in the topic on the y-axis
    """
    data = topic_novelty_df.query("year == @year").query("doc_counts >= @min_docs")

    fig = (
        alt.Chart(data)
        .mark_circle()
        .encode(
            # log scale for doc_counts
            x=alt.X(
                "doc_counts:Q",
                title="Popularity (number of documents)",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y(f"{novelty_measure}:Q", title=novelty_measure_label),
            tooltip=[
                alt.Tooltip("topic_name", title="Topic name"),
                alt.Tooltip(novelty_measure, title=novelty_measure_label),
                alt.Tooltip("doc_counts", title="Number of documents"),
            ],
        )
    )
    return fig


def chart_topic_novelty_patent_vs_openalex(
    topic_novelty_wide_df: pd.DataFrame,
) -> alt.Chart:
    """
    Chart show novelty for the same topics, for patents vs. OpenAlex, for a given year
    and indicate four quadrants depending on whether patents or OpenAlex are more novel

    Args:
        topic_novelty_wide_df (pd.DataFrame): Wide novelty table with columns:
            - topic: topic label
            - topic_name: topic name
            - openalex_zscore: z-score of novelty score for OpenAlex
            - patents_zscore: z-score of novelty score for patents

    Returns:
        alt.Chart: A scatter plot with novelty for patents on the x-axis and
            novelty for OpenAlex on the y-axis
    """
    fig = (
        alt.Chart(topic_novelty_wide_df)
        .mark_circle()
        .encode(
            x=alt.X("openalex_zscore:Q", title="OpenAlex novelty (z-score)"),
            y=alt.Y("patents_zscore:Q", title="Patent novelty (z-score)"),
            tooltip=[
                alt.Tooltip("topic_name:N", title="Topic title"),
                alt.Tooltip("openalex_zscore:Q", title="OpenAlex novelty (z-score)"),
                alt.Tooltip("patents_zscore:Q", title="Patent novelty (z-score)"),
            ],
        )
    )
    # add a mark for the origin
    line_vertical = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule().encode(x="x")
    line_horizontal = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule().encode(y="y")
    return fig + line_vertical + line_horizontal


def chart_topic_novelty_timeseries(
    topic_novelty_df: pd.DataFrame,
    topics_to_show: List[str],
    values_label: str = "Topic novelty",
    show_legend: bool = True,
) -> alt.Chart:
    """
    Chart showing topic novelty over time

    Args:
        topic_novelty_df (pd.DataFrame): Topic novelty dataframe with columns
            topic_name, topic_doc_novelty, doc_counts
        topics_to_show (List[str]): List of topics  (labels) show in the char
        values_label (str, optional): Label for the y-axis
        show_legend (bool, optional): Whether to show the legend

    Returns:
        alt.Chart: A line chart with topic novelty over time
            on the y-axis and year on the x-axis
    """

    data = topic_novelty_df.query("topic in @topics_to_show")
    legend = alt.Legend(orient="right") if show_legend else None
    return (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("year:O", title=""),
            y=alt.Y(f"{NOVELTY_MEASURE}:Q", title=values_label),
            color=alt.Color(
                "topic_name:N",
                title="Topic",
                scale=alt.Scale(scheme="tableau20"),
                legend=legend,
            ),
            tooltip=[
                alt.Tooltip("topic_name:N", title="Topic"),
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip(f"{NOVELTY_MEASURE}:Q", title=values_label, format=".3f"),
            ],
        )
    )


def chart_magnitude_growth(
    magnitude_growth_df: pd.DataFrame,
    growth: bool = True,
    show_doc_counts: bool = False,
) -> alt.Chart:
    """
    Chart showing the relationship between the magnitude and growth of topic novely over time

    Args:
        magnitude_growth_df (pd.DataFrame): Topic magnitude and growth dataframe with columns
            topic_name, avg_novelty, growth
        growth (bool, optional): Whether to show topics with growth or decline
        show_doc_counts (bool, optional): Whether to use the bubble size
            to visualise the number of documents for each topic

    Returns:
        alt.Chart: A scatter plot with topic novelty magnitude on the x-axis and
            growth on the y-axis
    """
    if growth:
        data = magnitude_growth_df.query("growth > 0")
        growth_label = "Growth"
        reverse_axis = False
    else:
        data = magnitude_growth_df.query("growth < 0").assign(
            growth=lambda x: x.growth.abs()
        )
        growth_label = "Decline"
        reverse_axis = True

    # add a mark for the median novelty
    median_novelty = magnitude_growth_df.avg_novelty.median()
    line_vertical = (
        alt.Chart(pd.DataFrame({"x": [median_novelty]})).mark_rule().encode(x="x")
    )

    # axis limits
    min_x = magnitude_growth_df.avg_novelty.min()
    max_x = magnitude_growth_df.avg_novelty.max()

    # show doc counts
    if show_doc_counts:
        size = alt.Size("doc_counts", title="Number of documents")
    else:
        size = alt.Size()

    fig = (
        alt.Chart(data)
        .mark_circle()
        .encode(
            x=alt.X(
                "avg_novelty",
                title="Average novelty",
                scale=alt.Scale(domain=(min_x, max_x)),
            ),
            y=alt.Y(
                "growth",
                title=growth_label,
                scale=alt.Scale(type="log", reverse=reverse_axis),
                axis=alt.Axis(format="%"),
            ),
            size=size,
            tooltip=[
                alt.Tooltip("topic_name", title="Topic"),
                alt.Tooltip("avg_novelty", title="Average novelty", format=".2f"),
                alt.Tooltip("growth", title=growth_label, format=".2%"),
                alt.Tooltip("doc_counts", title="Number of documents"),
            ],
        )
    )

    return fig + line_vertical