"""
forward citations pipeline
--------------------------

A pipeline that collects the citing works (via forward citations) for each
work collected using pipeline/data_collection/openalex.py.

This separates forward citation collection from works collection because the
total number of works involved in the citation graph is several times larger
than the core set of works and may generate a large volume of API calls.
"""
import os

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements_openalex_citations_flow.txt 1> /dev/null"
)


import boto3
import collections
import itertools
import json
import requests
import logging
import random
import sys
from typing import Dict, List

from metaflow import FlowSpec, step, Parameter, retry, batch, S3


API_ROOT = "https://api.openalex.org/works?filter=cites:"
PER_PAGE = 200
CALL_CHUNK_SIZE = 350
MAILTO = "data_analytics@nesta.org.uk"

TEST_YEAR = 2007
TEST_SIZE = 200
TEST_CALL_CHUNK_SIZE = 2


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_results(response: List[Dict]) -> Dict[str, List[str]]:
    """Parses OpenAlex API response to strip out all data except references.

    Args:
        response (List[Dict]): List of works and their metadata from OpenAlex
            API.

    Returns:
        outputs (Dict[str, List[str]]): Dictionary mapping work IDs to lists of
            work IDs that they cite.
    """
    outputs = dict()
    for result in response["results"]:
        outputs[result["id"]] = result["referenced_works"]

    return outputs


class OpenAlexForwardCitationsFlow(FlowSpec):
    production = Parameter(
        "production",
        help="Run in production?",
        default=False,
        type=bool,
    )
    focus_papers_only = Parameter(
        "focus_papers_only",
        help=(
            "If True, fetch only works that cite the focus papers and exclude "
            "works that cite the focus paper's references."
        ),
        default=True,
        type=bool,
    )
    start_year = Parameter(
        "start_year",
        help=(
            "Collect citations for focus papers published from, and including, "
            "this year."
        ),
        default=2007,
        type=int,
    )
    end_year = Parameter(
        "start_year",
        help=(
            "Collect citations for focus papers published until, and including, "
            "this year."
        ),
        default=2022,
        type=int,
    )
    min_citations = Parameter(
        "min_citations",
        help=(
            "The minimum number of citations a paper must have in order for "
            "it to be included in the search. Warning that this is based on "
            "data from get_openalex_works() which may be out of date. Use 0 "
            "to be safe (slower)."
        ),
        default=3,
        type=int,
    )

    @step
    def start(self):
        """Starts the flow."""
        self.next(self.generate_work_ids)

    @step
    def generate_work_ids(self):
        """Generates a list of focus paper work IDs to query from OpenAlex."""
        from dap_aria_mapping.getters.openalex import get_openalex_works

        s3 = boto3.resource("s3")

        if self.production:
            logger.info("Fetching core citations.")
            citations_object = s3.Object(
                "aria-mapping",
                f"inputs/data_collection/processed_openalex/citations.json",
            )
            if self.year is None:
                year_query = "publication_year > 0"
            else:
                year_query = "publication_year == @self.year"

            works = (
                get_openalex_works()
                .query("cited_by_count >= @self.min_citations")
                .query(year_query)["work_id"]
                .tolist()
            )
        else:
            logger.info(f"Fetching core citations for {TEST_YEAR}")
            citations_object = s3.Object(
                "aria-mapping",
                f"inputs/data_collection/processed_openalex/yearly_subsets/citations_{TEST_YEAR}.json",
            )

        citations_content = citations_object.get()["Body"].read().decode("utf-8")
        citations = json.loads(citations_content)
        if self.production:
            citations = {k: citations[k] for k in works}

        if self.focus_papers_only:
            all_work_ids = list(citations.keys())

        else:
            focus_work_ids = list(citations.keys())
            reference_work_ids = list(itertools.chain(*citations.values()))
            all_work_ids = set(focus_work_ids).union(set(reference_work_ids))

        random.shuffle(all_work_ids)

        if not self.production:
            all_work_ids = list(all_work_ids)[:TEST_SIZE]

        self.work_ids = [w.replace("https://openalex.org/", "") for w in all_work_ids]

        self.next(self.generate_calls)

    @step
    def generate_calls(self):
        """Generate chunks of API calls from work IDs."""
        import toolz

        # 50 is max number of parameters allowed in one OpenAlex call
        work_id_chunks = toolz.partition_all(50, self.work_ids)
        calls = [
            f"{API_ROOT}{'|'.join(work_ids)}?mailto={MAILTO}&per-page={PER_PAGE}"
            for work_ids in work_id_chunks
        ]

        chunk_size = CALL_CHUNK_SIZE if self.production else TEST_CALL_CHUNK_SIZE
        self.call_chunks = list(enumerate(toolz.partition_all(chunk_size, calls)))

        self.next(self.retrieve_data, foreach="call_chunks")

    @retry()
    @batch(cpu=2, memory=48_000)
    @step
    def retrieve_data(self):
        """Fetch response from API call chunks, parse and store."""
        i = self.input[0]
        logger.info(f"Collecting citations for chunk {i}")

        calls_to_make = len(self.input[1])
        calls_made = 0

        self.outputs = collections.defaultdict(list)
        for call in self.input[1]:
            cursor = "*"
            page_call = f"{call}&cursor={cursor}"
            resp = requests.get(page_call).json()

            total_results = resp["meta"]["count"]

            number_of_pages = -(total_results // -PER_PAGE)  # ceiling division

            self.outputs.update(parse_results(resp))
            cursor = resp["meta"]["next_cursor"]

            for _ in range(number_of_pages):
                page_call = f"{call}&cursor={cursor}"
                try:
                    resp = requests.get(page_call)
                    if resp.status_code == 200:
                        self.outputs.update(parse_results(resp.json()))
                        cursor = resp["meta"]["next_cursor"]
                    else:
                        raise ConnectionError(
                            f"Connection refused with error {resp.status_code}."
                        )
                except:
                    pass

            calls_made += 1
            logger.info(f"Made {calls_made} calls out of {calls_to_make}.")

        self.next(self.join)

    @step
    def join(self, inputs):
        """Joins all citation outputs and saves to S3."""
        import toolz

        citations_output = toolz.merge(*[input.outputs for input in inputs])

        fp = "focus_papers_only" if self.focus_papers_only else "expanded"

        if self.year is not None:
            fname = f"forward_citations_{fp}_{self.year}.json"
        else:
            fname = f"forward_citations_{fp}.json"

        if self.production:
            s3_path = f"inputs/data_collection/processed_openalex/{fname}"
        else:
            s3_path = f"inputs/data_collection/openalex/test/{fname}"

        s3 = boto3.resource("s3")
        s3object = s3.Object("aria-mapping", s3_path)
        s3object.put(Body=(bytes(json.dumps(citations_output).encode("UTF-8"))))

        self.next(self.end)

    @step
    def end(self):
        """Ends the flow."""
        pass


if __name__ == "__main__":
    OpenAlexForwardCitationsFlow()
