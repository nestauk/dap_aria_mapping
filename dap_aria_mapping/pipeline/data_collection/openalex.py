"""
works pipeline
--------------

A pipeline that takes a list of years, and outputs OpenAlex API results.

The thought behind this is to break the results into manageable yearly chunks. For a given year,
the output works may be well over 2GB in size when saved to json.
"""

import boto3
import io
import json
import requests

from metaflow import FlowSpec, S3, step, Parameter, retry, batch


<<<<<<< HEAD
YEARS = list(range(2007,2023))
=======
YEARS = [
    2007,
    2022,
    2021,
    2020,
    2019,
    2018,
    2017,
    2016,
    2015,
    2014,
    2013,
    2012,
    2011,
    2010,
    2009,
    2008,
]

>>>>>>> 2428cbd (Adding processing for raw openalex data)
API_ROOT = "https://api.openalex.org/works?filter="


def api_generator(api_root: str, year: int) -> list:
    """Generates a list of all URLs needed to completely collect
    all works relating to the list of concepts.

    Args:
        api_root : root URL of the OpenAlex API
        year : year to be queried against

    Returns:
        all_pages: list of pages required to return all results
    """
    page_one = f"{api_root}institutions.country_code:gb,publication_year:{year}"
    total_results = requests.get(page_one).json()["meta"]["count"]
    number_of_pages = -(total_results // -200)  # ceiling division
    all_pages = [
        f"{api_root}institutions.country_code:gb,publication_year:{year}&per-page=200&cursor="
        for _ in range(1, number_of_pages + 1)
    ]
    return all_pages


class OpenAlexWorksFlow(FlowSpec):
    production = Parameter("production", help="Run in production?", default=False)

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.generate_year_list)

    @step
    def generate_year_list(self):
        """Defines the years to be used"""
        # If production, generate all pages
        if self.production:
            self.year_list = YEARS
        else:
            self.year_list = YEARS[:1]
        self.next(self.retrieve_data, foreach="year_list")

    @retry()
    @batch(cpu=2, memory=48000)
    @step
    def retrieve_data(self):
        """Returns all results of the API hits"""
        # Get list of API calls
        api_call_list = api_generator(API_ROOT, self.input)
        # Get all results
        outputs = []
        cursor = "*"  # cursor iteration required to return >10k results
        for call in api_call_list:
            try: # catch transient errors
                req = requests.get(f"{call}{cursor}").json()
                for result in req["results"]:
                    outputs.append(result)
                cursor = req["meta"]["next_cursor"]
            except:
                pass
        # Define a filename and save to S3
        filename = f"openalex-gb-publications_year-{self.input}.json"
        obj = io.BytesIO(json.dumps(outputs).encode("utf-8"))
        s3 = boto3.client("s3")
        s3.upload_fileobj(obj, "aria-mapping", f"inputs/data_collection/openalex/{filename}")
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
    OpenAlexWorksFlow()
