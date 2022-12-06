import os

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements_openalex_flow.txt 1> /dev/null"
)

import json
from itertools import chain
from typing import Dict, List, Union
from metaflow import FlowSpec, step, Parameter, batch
import boto3
import pandas as pd
from toolz import pipe
import itertools


YEARS = list(range(2007, 2023))
OUTPUT_LOCATION = "inputs/data_collection/processed_openalex/yearly_subsets/"
INST_META_VARS = [
    "id",
    "display_name",
    "country_code",
    "type",
    "homepage_url"
]
WORK_META_VARS = [
    "id",
    "doi",
    "display_name",
    "publication_year",
    "publication_date",
    "cited_by_count",
    "is_retracted"
]
VENUE_META_VARS = [
    "id",
    "display_name",
    "url"
]


def deinvert_abstract(inverted_abstract: Dict[str, List]) -> Union[str, None]:
    """Convert inverted abstract into normal abstract

    Args:
        inverted_abstract: a dict where the keys are words
        and the values lists of positions

    Returns:
        A str that reconstitutes the abstract or None if the deinvered abstract
        is empty

    """

    if len(inverted_abstract) == 0:
        return None
    else:

        abstr_empty = (max(chain(*inverted_abstract.values())) + 1) * [""]

        for word, pos in inverted_abstract.items():
            for p in pos:
                abstr_empty[p] = word

        return " ".join(abstr_empty)

def extract_obj_meta(oalex_object: Dict, meta_vars: List) -> Dict:
    """Extracts variables of interest from an OpenAlex object (eg work, insitution...)

    Args:
        oalex_object: an OpenAlex object
        meta_vars: a list of variables to extract

    Returns:
        A dict with the variables of interest
    """

    return {var: val for var, val in oalex_object.items() if var in meta_vars}

def extract_work_venue(work: Dict, venue_vars: List) -> Dict:
    """Extracts nested metadata about a publication venue

    Args:
        work: an OpenAlex work
        venue_vars: a list of variables to extract

    Returns:
        A dict with the variables of interest
    """

    return {
        f"venue_{var}": val
        for var, val in work["host_venue"].items()
        if var in venue_vars
    }

def make_inst_metadata(inst_list: List, meta_vars: List) -> pd.DataFrame:
    """Makes a df with metadata about oalex institutions

    Args:
        doc_list: list of oalex institutions
        meta_vars: list of variables to extract

    Returns
        A df with instit-level metadata
    """

    return pipe(
        inst_list,
        lambda list_of_dicts: [
            extract_obj_meta(
                d,
                meta_vars=meta_vars,
            )
            for d in list_of_dicts
        ],
        pd.DataFrame,
    )

def make_work_metadata(work: Dict) -> Dict:
    """Extracts metadata about a work"""

    return {
        **extract_obj_meta(work, meta_vars=WORK_META_VARS),
        **extract_work_venue(work, venue_vars=VENUE_META_VARS),
    }

def make_work_corpus_metadata(works_list: List) -> pd.DataFrame:
    """Makes a df with work metadata

    Args:
        work_list: list of oalex works

    Returns:
        A df with work-level metadata
    """

    return pipe(
        works_list,
        lambda list_of_dicts: [make_work_metadata(pd.Series(d)) for d in list_of_dicts],
        pd.DataFrame,
    ).rename(columns={"id": "work_id"})

def get_nested_vars(work: Dict, variable: str, keys_to_keep: List) -> Union[None, List]:
    """Extracts nested variables from a document

    Args:
        doc: an open alex work
        nested_variable: the nested variable to extract
        keys_to_keep: the keys to keep in the nested variable

    Returns:
        A list of dicts with the nested variables
    """

    if variable not in work.keys():
        return None
    else:
        return [
            {
                **{"doc_id": work["id"]},
                **{k: v for k, v in conc.items() if k in keys_to_keep},
            }
            for conc in work[variable]
        ]

def make_work_concepts(
    works_list: List,
    variable: str = "concepts",
    keys_to_keep: List = ["id", "display_name", "score"],
    make_df: bool = True,
) -> pd.DataFrame:
    """
    Extracts concepts from work (could be openalex or mesh)

    Args:
        doc_list: list of openalex
        variable: concept variable to extract
        keys_to_keep: keys to keep in the concept
        make_df: whether to make a df or not

    Returns:
        A df with work-level concepts

    """

    return pipe(
        works_list,
        lambda doc_list: [
            get_nested_vars(doc, variable=variable, keys_to_keep=keys_to_keep)
            for doc in doc_list
        ],
        lambda dict_list: pd.DataFrame(chain(*dict_list))
        if make_df
        else chain(*dict_list),
    )

def get_authorships(work: Dict) -> List:
    """
    Extract authorships from a document

    Args:
        work: an openalex

    Returns:
        A list of parsed dicts with the authors and their affiliations
    """
    return list(
        chain(
            *[
                [
                    {
                        **{"id": work["id"]},
                        **{f"auth_{k}": v for k, v in auth["author"].items()},
                        **{"affiliation_string": auth["raw_affiliation_string"]},
                        **{f"inst_{k}": v for k, v in inst.items() if k == "id"},
                    }
                    for inst in auth[
                        "institutions"
                    ]  # Some authors are affiliated to more than
                    # one institution.
                ]
                for auth in work["authorships"]
            ]
        )
    )

def make_work_authorships(works_list: List) -> pd.DataFrame:
    """
    Creates a df with authors and institutions per works

    Args:
        works_list: list of openalex works
    """

    return pipe(
        works_list,
        lambda list_of_docs: [get_authorships(doc) for doc in list_of_docs],
        lambda extracted: pd.DataFrame(chain(*extracted)),
    )

def make_citations(work_list: List) -> Dict:
    """Dict with the papers cited by each work"""

    return {doc["id"]: doc["referenced_works"] for doc in work_list}


def make_deinverted_abstracts(work_list: List) -> Dict:
    """Dict with the deinverted abstracts of each work (where available"""

    return {
        doc["id"]: deinvert_abstract(doc["abstract_inverted_index"])
        if (type(doc["abstract_inverted_index"]) == dict)
        else None
        for doc in work_list
    }

class OpenAlexProcessedWorksFlow(FlowSpec):
    production = Parameter("production", help="Run in production?", default=False)

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.generate_years)

    @step
    def generate_years(self):
        """Defines the years to be used"""
        # If production, generate all pages
        if self.production:
            self.year_list = YEARS
        else:
            self.year_list = YEARS[:1]
        self.next(self.process_data, foreach="year_list")

    @batch(cpu=2, memory=24000)
    @step
    def process_data(self):
        # Fetch data
        s3 = boto3.resource("s3")
        content_object = s3.Object(
            "aria-mapping",
            f"inputs/data_collection/openalex/openalex-gb-publications_year-{self.input}.json"
        )
        file_content = content_object.get()["Body"].read().decode("utf-8")
        works = json.loads(file_content)
        # Works
        print("Processing works")
        works_df = make_work_corpus_metadata(works)
        s3_url = f"s3://aria-mapping/{OUTPUT_LOCATION}works_{self.input}.parquet"
        works_df.to_parquet(s3_url)
        # Concepts
        print("Processing concepts")
        concepts_df = make_work_concepts(works)
        s3_url = f"s3://aria-mapping/{OUTPUT_LOCATION}concepts_{self.input}.parquet"
        concepts_df.to_parquet(s3_url)
        # Authorships
        print("Processing authorships")
        authorships_df = make_work_authorships(works)
        s3_url = f"s3://aria-mapping/{OUTPUT_LOCATION}authorships_{self.input}.parquet"
        authorships_df.to_parquet(s3_url)
        # Citations
        citations_dict = make_citations(works)
        s3object = s3.Object("aria-mapping",
            f"{OUTPUT_LOCATION}citations_{self.input}.json")
        s3object.put(
            Body=(bytes(json.dumps(citations_dict).encode("UTF-8")))
        )
        # Deinverted abstracts
        deinverted_abstracts_dict = make_deinverted_abstracts(works)
        s3object = s3.Object("aria-mapping",
            f"{OUTPUT_LOCATION}abstracts_{self.input}.json")
        s3object.put(
            Body=(bytes(json.dumps(deinverted_abstracts_dict).encode("UTF-8")))
        )
        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Joins the outputs of the previous step
        """
        self.next(self.end)

    @step
    def end(self):
        """
        End of the flow, concatentates the outputs of the previous step
        """
        s3_resources = boto3.resource("s3")
        bucket = s3_resources.Bucket("aria-mapping")
        # Get all the keys in the bucket
        all_keys = [
            object_summary.key
            for object_summary
            in bucket.objects.filter(Prefix="inputs/data_collection/processed_openalex/yearly_subsets/")
        ]
        # Concatenate the parquet appropriate outputs
        for item in ["works", "authorships", "concepts"]:
            all_outputs = [
                pd.read_parquet(f"s3://aria-mapping/{output}")
                for output in all_keys if item in output
            ]
            all_dfs = pd.concat(all_outputs)
            s3_url = f"s3://aria-mapping/inputs/data_collection/processed_openalex/{item}.parquet"
            all_dfs.to_parquet(s3_url)
        # Concatenate the json appropriate outputs
        for item in ["citations", "abstracts"]:
            output_dict = {}
            all_outputs = [
                json.loads(s3_resources.Object("aria-mapping", output).get()["Body"].read().decode("utf-8"))
                for output in all_keys if item in output
            ]
            for output in all_outputs:
                output_dict.update(output)
            s3object = s3_resources.Object("aria-mapping",
                f"inputs/data_collection/processed_openalex/{item}.json")
            s3object.put(
                Body=(bytes(json.dumps(output_dict).encode("UTF-8")))
            )
            if item == "abstracts":
            # generate file in format {"id": id, "abstract": abstract} for annotation
                abstracts_for_annotation = [
                    {"id": key, "abstract": value}
                    for key, value in output_dict.items()
                ]
                s3object = s3_resources.Object("aria-mapping",
                    f"inputs/data_collection/processed_openalex/{item}_for_annotation.json")
                s3object.put(
                    Body=(bytes(json.dumps(abstracts_for_annotation).encode("UTF-8")))
                )


if __name__ == "__main__":
    OpenAlexProcessedWorksFlow()