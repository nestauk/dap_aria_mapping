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
from dap_aria_mapping.utils.patents import format_list_of_strings

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


def patents_citation_query(
    publication_numbers: List[str], production: bool = False
) -> str:
    """Returns a query that collects patent citation information for a list of focal_ids.

    Args:
        publication_numbers (str): List of publication numbers.
        production (bool, optional): If production, return query with full list of
            focal_ids, else return the first 10 elements of focal_ids.
            Defaults to False.

    Returns:
        str: Query string.
    """
    focal_ids = publication_numbers if production else publication_numbers[:10]

    focal_ids_formatted = format_list_of_strings(focal_ids)

    return f"select publication_number, citation from `patents-public-data.patents.publications` WHERE publication_number IN ({focal_ids_formatted})"


def patents_forward_citation_query(
    publication_numbers: List[str], production: bool = False
) -> str:
    """Returns a query that collects patent forward citation information
        for a list of focal_ids. i.e. for a given focal patent id, find the
        patents ids that cite them.

    Args:
        publication_numbers (List[str]): List of publication numbers.
        production (bool, optional): If production, return query with full list of
            focal_ids, else return the first 10 elements of focal_ids.
            Defaults to False.

    Returns:
        str: Query string.
    """
    focal_ids = publication_numbers if production else publication_numbers[:10]

    foward_citation_q = (
        f"WITH focal_id AS (SELECT * FROM UNNEST({focal_ids}) AS focal_ids)",
        "SELECT focal_id, p.publication_number AS cited_by",
        "FROM `patents-public-data.patents.publications` as p,",
        "UNNEST(p.citation) as citation, focal_id",
        "WHERE citation.publication_number IN (focal_ids)",
    )

    return " ".join(foward_citation_q)


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

        conn = est_conn()
        patents_citation_q = patents_citation_query(
            publication_numbers=self.publication_numbers, production=self.production
        )
        self.citation_results = conn.query(patents_citation_q).to_dataframe()

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

        self.next(self.retrieve_backward_citation_data)

    @step
    def retrieve_backward_citation_data(self):
        """Retrieve backward citation data from formatted citation data
        i.e. a list of patent ids that a given focal patent cites
        """
        self.backward_citations = (
            self.citations_data.groupby("focal_publication_number")
            .citation_publication_number.apply(list)
            .to_dict()
        )
        self.next(self.retrieve_foward_citation_data)

    @step
    def retrieve_foward_citation_data(self):
        """Retrieves forward citation data for focal ids in BigQuery
        i.e. for a given focal patent id, the patents ids
        that cite them.
        """
        from dap_aria_mapping.utils.conn import est_conn

        # I would self.conn in the previous step but it doesn't work with metaflow
        conn = est_conn()
        patents_backwards_citation_q = patents_forward_citation_query(
            publication_numbers=self.publication_numbers, production=self.production
        )
        results = conn.query(patents_backwards_citation_q).to_dataframe()

        results["focal_id"] = results.focal_id.apply(lambda x: x["focal_ids"])
        self.forward_citations = (
            results.groupby("focal_id").cited_by.apply(list).to_dict()
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
                self.backward_citations,
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
                "inputs/data_collection/patents/patents_citations_metadata.parquet",
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
