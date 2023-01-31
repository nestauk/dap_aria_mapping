"""
patent citations pipeline
--------------------------
A pipeline that collects the citing works (via forward, backward and citations metadata)
for each patent collected using pipeline/data_collection/patents.py.

python dap_aria_mapping/pipeline/data_collection/patents_citations.py run

NOTE: If the flow is in production, it will cost money to run this flow.
DO NOT run this flow unless you have explicitly confirmed with Genna or India.
"""

from metaflow import FlowSpec, step, Parameter
from typing import List
from dap_aria_mapping.utils.patents import format_list_of_strings, chunk

# from google patents data dictionary
CITATION_MAPPER = {
    "A": "technological background",
    "D": "document cited in application",
    "E": "earlier patent document",
    "1": "document cited for other reasons",
    "O": "Non-written disclosure",
    "P": "Intermediate document",
    "T": "theory or principle",
    "X": "relevant if taken alone",
    "Y": "relevant if combined with other documents",
    "CH2": "Chapter 2",
    "SUP": "Supplementary search report",
    "ISR": "International search report",
    "SEA": "Search report",
    "APP": "Applicant",
    "EXA": "Examiner",
    "OPP": "Opposition",
    "115": "article 115",
    "PRS": "Pre-grant pre-search",
    "APL": "Appealed",
    "FOP": "Filed opposition",
}

# from google bigquery docs
MAX_SIZE = 1024000


def base_patents_citation_query() -> str:
    """Returns a query that collects patent citation information for a list of focal_ids.

    Returns:
        str: Base query string.
    """
    return "SELECT publication_number, citation from `patents-public-data.patents.publications` WHERE publication_number IN ({})"


def base_patents_backward_citation_query() -> str:
    """Returns a base query that collects patent backward citation information
        for a list of focal_ids. i.e. for a given focal patent id, find the
        patents ids that cite them.

    Returns:
        str: Base query string.
    """

    return " ".join(
        (
            "WITH backward_citation AS (SELECT * FROM UNNEST([{}]) AS backward_citations)",
            "SELECT backward_citation, p.publication_number AS cited_by",
            "FROM `patents-public-data.patents.publications` as p,",
            "UNNEST(p.citation) as citation, backward_citation",
            "WHERE citation.publication_number IN (backward_citations)",
        )
    )


def chunk_bigquery_q(
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

        self.publication_numbers = get_patents().publication_number.tolist()
        self.next(self.retrieve_citation_data)

    @step
    def retrieve_citation_data(self):
        """Retrieves citation data from BigQuery"""
        from dap_aria_mapping.utils.conn import est_conn
        import pandas as pd

        conn = est_conn()
        base_citations_q = base_patents_citation_query()
        base_citation_q_chunks = chunk_bigquery_q(
            self.publication_numbers,
            base_q=base_citations_q,
            production=self.production,
        )
        if len(base_citation_q_chunks) == 1:
            self.citation_results = conn.query(base_citation_q_chunks[0]).to_dataframe()
        else:
            self.citation_results = pd.concat(
                [conn.query(q).to_dataframe() for q in base_citation_q_chunks]
            )

        self.next(self.format_citation_data)

    @step
    def format_citation_data(self):
        """Formats citation data from BiqQuery"""
        import pandas as pd

        citation_df = pd.concat(
            [pd.DataFrame(i) for i in self.citation_results["citation"]],
            keys=self.citation_results.index,
        ).reset_index(level=1, drop=True)
        citation_df_exploded = citation_df[0].apply(pd.Series)

        self.citations_data = (
            pd.merge(
                self.citation_results,
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

        self.next(self.retrieve_forward_citation_data)

    @step
    def retrieve_forward_citation_data(self):
        """Retrieve backward citation data from formatted citation data
        i.e. a list of patent ids that a given focal patent cites
        """
        self.forward_citations = (
            self.citations_data.groupby("focal_publication_number")
            .citation_publication_number.apply(list)
            .to_dict()
        )
        self.next(self.retrieve_backward_citation_data)

    @step
    def retrieve_backward_citation_data(self):
        """Retrieves backward citation data for forward citations in BigQuery
        i.e. for a given forward citation of a focal patent id, the patents ids
        that cite them.
        """
        from dap_aria_mapping.utils.conn import est_conn
        import itertools
        import pandas as pd

        forward_citation_ids = list(
            set(itertools.chain(*list(self.forward_citations.values())))
        )
        # I would self.conn in the previous step but it doesn't work with metaflow
        conn = est_conn()
        base_backward_citations_q = base_patents_backward_citation_query()
        base_citation_q_chunks = chunk_bigquery_q(
            forward_citation_ids,
            base_q=base_backward_citations_q,
            production=self.production,
        )
        if len(base_backward_citations_q) == 1:
            backward_citation_results = conn.query(
                base_citation_q_chunks[0]
            ).to_dataframe()
        else:
            backward_citation_results = pd.concat(
                [conn.query(q).to_dataframe() for q in base_citation_q_chunks]
            )

        self.backward_citation_dict = (
            backward_citation_results.assign(
                backward_citation=lambda x: x.backward_citation.apply(
                    lambda x: x["backward_citations"]
                )
            )
            .groupby("backward_citation")
            .cited_by.apply(list)
            .to_dict()
        )

        self.next(self.save_data)

    @step
    def save_data(self):
        """Saves forward, backward and citation metadata as
        json/parquet files to s3"""
        from nesta_ds_utils.loading_saving.S3 import upload_obj
        from dap_aria_mapping import BUCKET_NAME

        if self.production:
            upload_obj(
                self.backward_citation_dict,
                BUCKET_NAME,
                "inputs/data_collection/patents/patents_backward_citations.json",
            )
            upload_obj(
                self.forward_citations,
                BUCKET_NAME,
                "inputs/data_collection/patents/patents_forward_citations.json",
            )
            upload_obj(
                self.citations_data,
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
