"""
patent citations pipeline
--------------------------
A pipeline that collects the citing works (via forward and backward citations)
for each patent collected using pipeline/data_collection/patents.py.

python dap_aria_mapping/pipeline/data_collection/patents_citations.py run

NOTE: If the flow is in production, it will cost money to run this flow.
DO NOT run this flow unless you have explicitly confirmed with Genna or India.
"""
from typing import List, Dict, Tuple

from metaflow import FlowSpec, step, Parameter
import pandas as pd
from google.cloud import bigquery

from dap_aria_mapping.utils.patents import (
    format_list_of_strings,
    chunk,
    CITATION_MAPPER,
    MAX_SIZE,
    EARLY_YEAR,
    LATE_YEAR,
)

from dap_aria_mapping.utils.conn import est_conn

# create connection to bigquery
CONN = est_conn()


def base_patents_backward_citation_query() -> str:
    """Returns a query that collects patent citation information
        (i.e. 'patent citations' in Google Patents front end. AKA backward citations)
        for a list of focal_ids.

    Returns:
        str: Base query string.
    """
    return "SELECT publication_number, citation from `patents-public-data.patents.publications` WHERE publication_number IN ({})"


def base_patents_forward_citation_query() -> str:
    """Returns a query that collects patent forward citation information
    (i.e. 'cited by' in Google Patents front end. AKA frontward citations)
    for a list of focal_ids.

    Returns:
        str: Base query string.
    """
    return "SELECT c.publication_number as focal_id, p.publication_number as cited_by from `patents-public-data.patents.publications` as p, unnest(citation) as c WHERE c.publication_number IN ({})"


def chunk_bigquery_query(
    publication_numbers: List[str], base_q: str, production: bool = False
) -> List[List[str]]:
    """Chunks a bigquery query into smaller queries to accomodate
        maximum query string length.

    Args:
        publication_numbers (List[str]): List of patent publication numbers.
        base_q (str): base bigquery query string.
        production (bool, optional): If in production or not. Defaults to False.

    Returns:
        List[List[str]]: List of bigquery query strings.
    """

    ids = publication_numbers if production else publication_numbers[:10]

    max_q_size = MAX_SIZE - len(base_q)
    publication_numbers_char_size = len(", ".join(ids))

    chunk_size = len([i for i in range(0, publication_numbers_char_size, max_q_size)])

    publication_numbers_chunks = chunk(ids, chunk_size)

    chunk_qs = []
    for pub_chunk in publication_numbers_chunks:
        chunk_formatted = format_list_of_strings(pub_chunk)
        chunk_qs.append(base_q.format(chunk_formatted))

    return chunk_qs


def format_citation_data(
    citation_results: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Formats citation data into a dataframe and a dictionary.

    Args:
        citation_results (pd.DataFrame): BigQuery citation results.

    Returns:
        Tuple[pd.DataFrame, Dict[str, List[str]]]: Formatted citation dataframe and citations dictionary.
    """

    citation_df = pd.concat(
        [pd.DataFrame(i) for i in citation_results["citation"]],
        keys=citation_results.index,
    ).reset_index(level=1, drop=True)
    citation_df_exploded = citation_df[0].apply(pd.Series)

    citations_data = (
        pd.merge(
            citation_results,
            citation_df_exploded,
            left_index=True,
            right_index=True,
        )
        .drop(columns=["citation"])
        .rename(
            columns={
                "publication_number_x": "focal_publication_number",
                "publication_number_y": "citation_publication_number",
            }
        )
        .reset_index(drop=True)
        .assign(type_description=lambda x: x.type.map(CITATION_MAPPER))
        .assign(category_description=lambda x: x.category.map(CITATION_MAPPER))
        .query("citation_publication_number != ''")
    )

    citations_data_dict = (
        citations_data.groupby("focal_publication_number")
        .citation_publication_number.apply(list)
        .to_dict()
    )

    return citations_data, citations_data_dict


def get_citations(
    query: str,
    publication_numbers: List[str],
    conn: bigquery.Client = CONN,
    production: bool = False,
) -> pd.DataFrame:
    """Retrieves citation data from Google BigQuery. If query is too large,
        it will be chunked into smaller queries. Results are concatenated.

    Args:
        query (str): Base query string.
        publication_numbers (List[str]): List of publication numbers.
        conn (bigquery.Client, optional): BigQuery client. Defaults to CONN.
        production (bool, optional): If in production. Defaults to False.

    Returns:
        pd.DataFrame: Citation data.
    """
    base_citation_q_chunks = chunk_bigquery_query(
        publication_numbers,
        base_q=query,
        production=production,
    )
    if len(base_citation_q_chunks) == 1:
        citation_results = conn.query(base_citation_q_chunks[0]).to_dataframe()
    else:
        citation_results = pd.concat(
            [conn.query(q).to_dataframe() for q in base_citation_q_chunks]
        )

    return citation_results


class PatentsCitationsFlow(FlowSpec):
    production = Parameter("production", help="Run in production?", default=False)

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.get_focal_ids)

    @step
    def get_focal_ids(self):
        """Gets list of focal_ids from patents data"""
        from dap_aria_mapping.getters.patents import get_patents

        # get list of publication numbers i.e. focal ids from patents data for early and late years
        self.publication_numbers = (
            get_patents()
            .assign(publication_date=lambda x: pd.to_datetime(x.publication_date))
            .assign(publication_year=lambda x: x.publication_date.dt.year)
            .pipe(lambda x: x[x.publication_year.isin([EARLY_YEAR, LATE_YEAR])])
            .publication_number.tolist()
        )

        self.next(self.retrieve_backward_citation_data)

    @step
    def retrieve_backward_citation_data(self):
        """Retrieves backward citation data for a list of initial focal ids from BigQuery"""

        # generate bigquery query to retrieve backward citations based on focal ids
        backward_citations_q = base_patents_backward_citation_query()

        # call BigQuery API using backward_citations_q to retrieve backward citations
        backward_citations = get_citations(
            backward_citations_q, self.publication_numbers, production=self.production
        )
        # format backward citations into dataframe and dictionary where key is focal id and value is list of backward citations (or 'patent citations' in Google Patents front end)
        (
            self.backward_citations_clean,
            self.backward_citations_dict,
        ) = format_citation_data(backward_citations)

        self.next(self.retrieve_cited_by_data)

    @step
    def retrieve_cited_by_data(self):
        """Retrieves 'cited by' patent ids from list of focal ids."""
        # generate bigquery query to retrieve 'cited by' patent ids based on focal ids
        cited_by_query = base_patents_forward_citation_query()

        # call BigQuery API using cited_by_query to retrieve 'cited by' patent ids
        cited_by_results = get_citations(
            cited_by_query, self.publication_numbers, production=self.production
        )

        # generate dictionary of 'cited by' patent ids where key is focal id and value is list of 'cited by' patent ids
        self.cited_by_results_dict = (
            cited_by_results.groupby("focal_id").cited_by.apply(list).to_dict()
        )

        self.next(self.retrieve_forward_citation_data)

    @step
    def retrieve_forward_citation_data(self):
        """Using the 'cited by' patent ids, retrieve the 'cites' patent ids to generate forward citations."""
        import itertools

        # flatten list of 'cited by' patent ids
        flat_cited_by_results = list(
            set(itertools.chain(*list(self.cited_by_results_dict.values())))
        )

        # generate bigquery query to retrieve 'cites' patent ids for 'cited by' patent ids
        base_forward_q = base_patents_backward_citation_query()

        # call BigQuery API using base_forward_q and flat_cited_by_results
        forward_citations = get_citations(
            base_forward_q, flat_cited_by_results, production=self.production
        )

        # clean results and generate forward citations dictionary
        (
            self.forward_citations_clean,
            self.forward_citations_dict,
        ) = format_citation_data(forward_citations)

        self.next(self.save_data)

    @step
    def save_data(self):
        """Saves forward and backward citations as json/parquet files to s3"""
        from nesta_ds_utils.loading_saving.S3 import upload_obj
        from dap_aria_mapping import BUCKET_NAME

        if self.production:
            upload_obj(
                self.backward_citation_dict,
                BUCKET_NAME,
                "inputs/data_collection/patents/patents_backward_citations.json",
            )
            upload_obj(
                self.backward_citations_clean,
                BUCKET_NAME,
                "inputs/data_collection/patents/patents_backward_citations_metadata.parquet",
            )
            upload_obj(
                self.forward_citations_dict,
                BUCKET_NAME,
                "inputs/data_collection/patents/patents_forward_citations.json",
            )
            upload_obj(
                self.forward_citations_clean,
                BUCKET_NAME,
                "inputs/data_collection/patents/patents_forward_citations_metadata.parquet",
            )
        else:
            pass

        self.next(self.end)

    @step
    def end(self):
        """Ends the flow"""
        pass


if __name__ == "__main__":
    PatentsCitationsFlow()
