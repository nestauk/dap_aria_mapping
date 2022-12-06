"""
patents pipeline
--------------
A pipeline that queries patents from Google Bigquery and
saves patents results as a parquet file to s3.

The pipeline collect all patents, deduplicated on family ID, of
inventors based in the UK, filed between 2016 and 2021.

python dap_aria_mapping/pipeline/data_collection/patents.py run

NOTE: If the flow is in production, it will cost money to run this flow.
DO NOT run this flow unless you have explicitly confirmed with Genna or India.
"""
from metaflow import FlowSpec, step, Parameter


def patents_query(production=False) -> str:
    """Generates patents query that collects information on
        patents, deduplicated on family ID, of inventors based
        in the UK filed between 2016 and 2021.

        If production is false, query for 10 publication numbers,
            else query for all publication numbers.

    Returns query string.
    """
    filter_q = (
        "select ANY_VALUE(publication_number) publication_number ",
        "from `patents-public-data.patents.publications`, ",
        "unnest(inventor_harmonized) as inventor, unnest(abstract_localized) as abstract ",
        "where cast(filing_date as string) between '20160101' and '20211231' ",
        "and inventor.country_code = 'GB' and abstract.language = 'en' ",
        "GROUP BY family_id",
    )
    columns_q = (
        "publication_number, application_number, country_code, ",
        "family_id, title_localized, abstract_localized, ",
        "publication_date, filing_date, grant_date, priority_date, cpc, ",
        "inventor_harmonized, assignee_harmonized",
    )
    q = "".join(
        (
            f"with filtered_ids as ({''.join(filter_q)}) select {''.join(columns_q)} ",
            "from `patents-public-data.patents.publications` INNER JOIN filtered_ids USING(publication_number)",
        )
    )

    if production:
        return q
    else:
        return f"{q} LIMIT 10"


class PatentsFlow(FlowSpec):
    production = Parameter("production", help="Run in production?", default=True)

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.retrieve_data)

    @step
    def retrieve_data(self):
        """Retrieves relevant data from BigQuery"""
        from dap_aria_mapping.utils.conn import est_conn

        conn = est_conn()
        patents_q = patents_query(production=self.production)
        self.results = conn.query(patents_q).to_dataframe()
        self.next(self.save_data)

    @step
    def save_data(self):
        """Saves data as parquet file to s3"""
        from nesta_ds_utils.loading_saving.S3 import upload_obj
        from dap_aria_mapping import BUCKET_NAME

        upload_obj(
            self.results,
            BUCKET_NAME,
            "inputs/data_collection/patents/patents_raw.parquet",
        )

        self.next(self.end)

    @step
    def end(self):
        """Ends the flow"""
        pass


if __name__ == "__main__":
    PatentsFlow()
