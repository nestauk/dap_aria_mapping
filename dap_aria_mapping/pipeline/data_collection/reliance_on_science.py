"""
reliance on science pipeline
--------------

A pipeline that collects the reliance on science dataset
"""

import pandas as pd

from metaflow import FlowSpec, S3, step, Parameter, retry, batch

URL = "https://zenodo.org/record/7497435/files/_pcs_mag_doi_pmid.tsv"


class RelianceOnScienceFlow(FlowSpec):
    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.download_data)

    @step
    def download_data(self):
        """Downloads the data"""
        self.data = pd.read_csv(URL, sep="\t")
        self.next(self.save_to_s3)

    @step
    def save_to_s3(self):
        """Saves the data to S3"""
        s3_url = "s3://aria-mapping/inputs/data_collection/reliance_on_science/reliance_on_science.parquet"
        self.data.to_parquet(s3_url)
        self.next(self.end)

    @step
    def end(self):
        """Ends the flow"""
        pass


if __name__ == "__main__":
    RelianceOnScienceFlow()
