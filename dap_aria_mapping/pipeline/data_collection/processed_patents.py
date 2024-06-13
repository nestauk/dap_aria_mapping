"""
process patents pipeline
--------------
A pipeline that processes fetched patents from Google Bigquery and
saves processed patents results as a parquet file to s3.

It also generates and saves a patents look up file to be used for entity
extraction.

python dap_aria_mapping/pipeline/data_collection/processed_patents.py run
"""
import pandas as pd
from typing import List, Union
from metaflow import FlowSpec, step, Parameter
from dap_aria_mapping import BUCKET_NAME, logger

DATE_COLS = ["publication_date", "filing_date", "grant_date", "priority_date"]
TEXT_COLS = ["title_localized", "abstract_localized"]
COLS_TO_UNNEST = ["inventor_harmonized", "assignee_harmonized"]


def unnest_column(nested_col: str) -> Union[List[str], List[str]]:
    """Unnests country code and name from nested column
    of list of dictionaries with keys 'country_code' and 'name'
    """
    country_code = [col["country_code"] for col in nested_col]
    name = [col["name"] for col in nested_col]

    return country_code, name


def extract_english_text(text_col: str) -> str:
    """Extracts english text from text columns"""
    for text in text_col:
        if text["language"] == "en":
            return text["text"]


class PatentsProcessedFlow(FlowSpec):
    raw_patents_path = Parameter(
        "raw_patents_path",
        help="s3 location of raw patents path",
        default="inputs/data_collection/patents/patents_raw.parquet",
    )

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.load_data)

    @step
    def load_data(self):
        """
        Loads raw patents parquet file from S3
        """
        from nesta_ds_utils.loading_saving.S3 import download_obj

        logger.info(f"loading {self.raw_patents_path} from s3...")

        self.patents = download_obj(
            BUCKET_NAME, self.raw_patents_path, download_as="dataframe"
        )
        self.next(self.convert_dates)

    @step
    def convert_dates(self):
        """
        Converts all dates in patents data to '%Y%m%d' format
        """
        for date_col in DATE_COLS:
            self.patents[date_col] = pd.to_datetime(
                self.patents[date_col], format="%Y%m%d", errors="coerce"
            )
        self.next(self.clean_texts)

    @step
    def clean_texts(self):
        """
        Extracts english text in patents text columns
        """
        for text_col in TEXT_COLS:
            self.patents[text_col] = self.patents[text_col].apply(extract_english_text)
        self.next(self.unnest_columns)

    @step
    def unnest_columns(self):
        """
        Unnest columns with dictionaries that contain
            'country code' and 'name' keys
        """
        for unnest_col in COLS_TO_UNNEST:
            self.patents[unnest_col] = self.patents[unnest_col].apply(unnest_column)
            (
                self.patents[f"{unnest_col}_country_codes"],
                self.patents[f"{unnest_col}_names"],
            ) = zip(*self.patents[unnest_col])
        self.next(self.extract_cpc_codes)

    @step
    def extract_cpc_codes(self):
        """
        Extract all CPC codes
        """
        self.patents["cpc"] = self.patents["cpc"].apply(
            lambda x: [c["code"] for c in x]
        )
        self.next(self.save_data)

    # ignore this step for collecting additional patents data

    # @step
    # def generate_lookup(self):
    #     """
    #     Create patent abstract look up for entity extraction
    #     """
    #     self.patents["abstract_localized"] = self.patents[
    #         "abstract_localized"
    #     ].str.replace("\n", "")
    #     self.patents_lookup = list(
    #         self.patents.rename(
    #             columns={"publication_number": "id", "abstract_localized": "abstract"}
    #         )[["id", "abstract"]]
    #         .T.to_dict()
    #         .values()
    #     )
    #     self.next(self.save_data)

    @step
    def save_data(self):
        """Saves clean patents data as parquet file to s3

        Also saves patents look up as json to s3 for
            entity extraction
        """
        from nesta_ds_utils.loading_saving.S3 import upload_obj

        clean_patents_path = self.raw_patents_path.replace("raw", "clean")
        # annotation_patents_path = self.raw_patents_path.replace('raw', 'for_annotation')

        upload_obj(
            self.patents,
            BUCKET_NAME,
            clean_patents_path,
        )

        # upload_obj(
        #     self.patents_lookup,
        #     BUCKET_NAME,
        #     annotation_patents_path,
        # )

        self.next(self.end)

    @step
    def end(self):
        """Ends the flow"""
        pass


if __name__ == "__main__":
    PatentsProcessedFlow()
