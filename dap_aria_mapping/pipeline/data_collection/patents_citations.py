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
)

from dap_aria_mapping.utils.conn import est_conn
from dap_aria_mapping import logger
from toolz import partition_all

# of patent ids to include in chunk
CHUNK_SIZE = 10000

# create connection to bigquery
conn = est_conn()


def patents_backward_citation_query() -> str:
    """Returns a query that collects patent citation information
        (i.e. 'patent citations' in Google Patents front end)
        for a list of focal_ids.

    Returns:
        str: Base query string.
    """
    return """SELECT p.publication_number as focal_id, c.publication_number
            as patent_citation from `patents-public-data.patents.publications` as p,
            unnest(citation) c WHERE p.publication_number IN ({})"""


def patents_forward_citation_query() -> str:
    """Returns a query that collects patent forward citation information
    (i.e. 'cited by' in Google Patents front end)
    for a list of focal_ids.

    Returns:
        str: Base query string.
    """

    return """SELECT c.publication_number as focal_id, p.publication_number as cited_by
    from `patents-public-data.patents.publications` as p,
    unnest(citation) c WHERE c.publication_number IN ({})"""


def chunk_bigquery_query(
    publication_numbers: List[str],
    base_q: str,
    production: bool = False,
    chunk_size: int = CHUNK_SIZE,
) -> List[List[str]]:
    """Chunk bigquery query into chunk_size number of chunks

    Args:
        publication_numbers (List[str]): List of publication numbers
        chunk_size (int): Number of ids to include in each chunk
        base_q (str): Base query string
        production (bool, optional): If query is in production. Defaults to False.

    Returns:
        List[List[str]]: List of query strings
    """
    ids = publication_numbers if production else publication_numbers[:10]

    publication_numbers_chunks = list(partition_all(chunk_size, ids))

    return [
        base_q.format(format_list_of_strings(pub_chunk))
        for pub_chunk in publication_numbers_chunks
    ]


class PatentsCitationsFlow(FlowSpec):
    production = Parameter("production", help="Run in production?", default=False)
    chunk_size = Parameter(
        "chunk_size", help="# of BigQuery queries", default=CHUNK_SIZE
    )
    year = Parameter("year", help="year to collect citations data for", default=2007)

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
        logger.info(f"Fetching list of focal ids for patents published in {self.year}.")

        publication_numbers = (
            get_patents()
            .assign(publication_date=lambda x: pd.to_datetime(x.publication_date))
            .assign(publication_year=lambda x: x.publication_date.dt.year)
            .pipe(lambda x: x[x.publication_year == self.year])
            .publication_number.tolist()
        )

        self.publication_numbers = (
            publication_numbers if self.production else publication_numbers[:CHUNK_SIZE]
        )

        self.publication_numbers_formatted = format_list_of_strings(
            self.publication_numbers
        )

        logger.info(
            f"Retrieving forward and backward citation data for {len(self.publication_numbers)} focal ids."
        )

        self.next(self.retrieve_backward_citation_data)

    @step
    def retrieve_backward_citation_data(self):
        """Retrieves backward citation data for a list of initial focal ids from BigQuery"""
        from nesta_ds_utils.loading_saving.S3 import upload_obj
        from dap_aria_mapping import BUCKET_NAME

        logger.info(f"Retrieving backward citation data for focal ids.")

        # generate bigquery query to retrieve backward citations based on focal ids
        backward_citations_q = patents_backward_citation_query().format(
            self.publication_numbers_formatted
        )

        # call BigQuery API using backward_citations_q to retrieve backward citations
        citation_results = conn.query(backward_citations_q).to_dataframe()

        # format backward citations into dictionary where key is focal id and value is list of backward citations (or 'patent citations' in Google Patents front end)
        backward_citations = (
            citation_results.groupby("focal_id").patent_citation.apply(list).to_dict()
        )

        # upload backward citations to s3
        if self.production:
            logger.info("saving backward citation data to s3")
            upload_obj(
                backward_citations,
                BUCKET_NAME,
                f"inputs/data_collection/patents/yearly_subsets/patents_backward_citations_{str(self.year)}.json",
            )

        self.next(self.retrieve_cited_by_data)

    @step
    def retrieve_cited_by_data(self):
        """Retrieves 'cited by' data for a list of initial focal ids from BigQuery"""
        from nesta_ds_utils.loading_saving.S3 import upload_obj
        from dap_aria_mapping import BUCKET_NAME

        logger.info(f"Retrieving 'cited by' patent ids for focal ids.")

        # generate bigquery query to retrieve cited by citations based on focal ids
        cited_by_q = patents_forward_citation_query().format(
            self.publication_numbers_formatted
        )

        # call BigQuery API using cited_by_q to retrieve cited by citations
        cited_by_results = conn.query(cited_by_q).to_dataframe()

        # generate dictionary of 'cited by' patent ids where key is focal id and value is list of 'cited by' patent ids
        self.cited_by_results_dict = (
            cited_by_results.groupby("focal_id").cited_by.apply(list).to_dict()
        )

        # don't know if this is needed but just in case - save cited by data to s3
        if self.production:
            logger.info("saving backward citation data to s3")
            upload_obj(
                self.cited_by_results_dict,
                BUCKET_NAME,
                f"inputs/data_collection/patents/yearly_subsets/patents_cited_by_citations_{str(self.year)}.json",
            )

        self.next(self.generate_forward_citation_query_chunks)

    @step
    def generate_forward_citation_query_chunks(self):
        """Generates chunks of forward citation queries to be run in parallel"""
        import itertools

        logger.info(
            f"Generate forward citation chunks with {self.chunk_size} patent ids each."
        )

        # flatten list of 'cited by' patent ids
        flat_cited_by_results = list(
            set(itertools.chain(*list(self.cited_by_results_dict.values())))
        )

        # we want the backward citations of the 'cited by' patents as the forward citations of the focal patents
        base_q = patents_backward_citation_query()
        # generate chunk_size number of query chunks to retrieve forward citation data
        self.query_chunks = chunk_bigquery_query(
            publication_numbers=flat_cited_by_results,
            chunk_size=self.chunk_size,
            base_q=base_q,
            production=self.production,
        )

        self.next(self.retrieve_forward_citation_data, foreach="query_chunks")

    @step
    def retrieve_forward_citation_data(self):

        logger.info(f"Retrieving forward citation data for focal ids.")
        # query bigquery to retrieve forward citation data in chunks
        self.forward_citation_chunks = conn.query(self.input).to_dataframe()

        self.next(self.join)

    @step
    def join(self, inputs):
        """Join and clean forward citation results"""

        # join forward citation results
        logger.info(f"join {self.chunk_size} forward citation results")
        forward_citation_results = pd.concat(
            [i.forward_citation_chunks for i in inputs]
        )

        logger.info(f"{forward_citation_results.shape}")

        # generate forward citations dictionary
        self.forward_citations = (
            forward_citation_results.groupby("focal_id")
            .patent_citation.apply(list)
            .to_dict()
        )

        self.next(self.end)

    @step
    def end(self):
        """saves forward citations to s3"""
        from nesta_ds_utils.loading_saving.S3 import upload_obj
        from dap_aria_mapping import BUCKET_NAME

        # save forward citations to s3
        if self.production:
            logger.info("saving forward citation data to s3")
            upload_obj(
                self.forward_citations,
                BUCKET_NAME,
                f"inputs/data_collection/patents/yearly_subsets/patents_forward_citations_{self.year}.json",
            )


if __name__ == "__main__":
    PatentsCitationsFlow()
