"""
patents pipeline
--------------
A pipeline that queries patents from Google Bigquery and
saves patents results as a parquet file to s3.

The pipeline collect all patents, deduplicated on family ID, of
inventors based in countries for a country code, filed between 2007 and 2023.

python dap_aria_mapping/pipeline/data_collection/patents.py run

NOTE: If the flow is in production, it will cost money to run this flow.
DO NOT run this flow unless you have explicitly confirmed with Genna or India.
"""
from metaflow import FlowSpec, step, Parameter
from typing import List
from dap_aria_mapping import logger

# Europe country codes from: https://docs.oracle.com/cd/E90599_01/html/Payments_Core/Payments_Maintenance_Annexure.htm
COUNTRY_CODES = [
    "CY",
    "ES",
    "SE",
    "IT",
    "DE",
    "FR",
    "EE",
    "GR",
    "HR",
    "LV",
    "MT",
    "LU",
    "LT",
    "NO",
    "LI",
    "IS",
    "SI",
    "NL",
    "AT",
    "PL",
    "FI",
    "DK",
    "BE",
    "PT",
    "RO",
    "CZ",
    "SK",
    "HU",
    "IE",
    "BG",
]


def patents_query(country_code: str, production=False) -> str:
    """Generates patents query that collects information on
        patents, deduplicated on family ID, of inventors based
        in a given region filed between 2007 and 2023.

        If production is false, query for 10 publication numbers,
            else query for all publication numbers.

    Returns query string.
    """

    filter_q = (
        "select ANY_VALUE(publication_number) publication_number ",
        "from `patents-public-data.patents.publications`, ",
        "unnest(inventor_harmonized) as inventor, unnest(abstract_localized) as abstract ",
        'where cast(filing_date as string) between "20070101" and "20221231" ',
        f'and inventor.country_code = "{country_code}" and abstract.language = "en" ',
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
    country_codes = Parameter(
        "country_codes", help="Country codes to be used", default=COUNTRY_CODES
    )

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.generate_country_code_list)

    @step
    def generate_country_code_list(self):
        """Defines the country codes to be used"""
        # If production, generate all pages
        if self.production:
            self.country_code_list = self.country_codes
        else:
            self.country_code_list = self.country_codes[:1]

        self.next(self.retrieve_data, foreach="country_code_list")

    @step
    def retrieve_data(self):
        """Retrieves relevant data from BigQuery and saves to s3"""
        from dap_aria_mapping.utils.conn import est_conn
        from nesta_ds_utils.loading_saving.S3 import upload_obj
        from dap_aria_mapping import BUCKET_NAME

        conn = est_conn()
        patents_q = patents_query(country_code=self.input, production=self.production)
        self.results = conn.query(patents_q).to_dataframe()

        logger.info(
            f"Retrieved {len(self.results)} patents for patents with at least one inventor from {self.input}."
        )

        data_path = f"inputs/data_collection/patents/eu_us_patents/patents_raw_{self.input}.parquet"

        if self.production:
            upload_obj(
                self.results,
                BUCKET_NAME,
                data_path,
            )
        else:
            pass

        self.next(self.dummy_join)

    @step
    def dummy_join(self, inputs):
        """Dummy join step"""
        self.next(self.end)

    @step
    def end(self):
        """Ends the flow"""
        pass


if __name__ == "__main__":
    PatentsFlow()
