# %%
from metaflow import (
    FlowSpec,
    S3,
    step,
    conda,
    Parameter,
    retry,
    batch,
    Run,
    project,
    current,
)
from metaflow.client.core import MetaflowData
from dap_aria_mapping import BUCKET_NAME
from toolz import pipe
from copy import deepcopy
from itertools import chain
import functools

from dap_aria_mapping.utils.semantics import get_sample, filter_entities


def pip(libraries):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            import subprocess
            import sys

            for library, version in libraries.items():
                print("Pip Install:", library, version)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        library + "==" + version,
                    ]
                )
            return function(*args, **kwargs)

        return wrapper

    return decorator


MODEL_NAME = "sentence-transformers/allenai-specter"

# %%
class GetEmbeddings(FlowSpec):
    test_mode = Parameter(
        "test",
        help="Determines whether the flow runs on a small set of articles",
        type=bool,
        default=True,
    )

    @step
    def start(self):
        from dap_aria_mapping.getters.openalex import get_openalex_entities
        from dap_aria_mapping.getters.patents import get_patent_entities

        self.oa_entities = get_openalex_entities()
        self.pat_entities = get_patent_entities()
        self.next(self.sample)

    @step
    def sample(self):
        num_articles = 1_000 if self.test_mode else -1
        self.oa_sample_data = get_sample(
            self.oa_entities, score_threshold=80, num_articles=num_articles
        )
        self.pat_sample_data = get_sample(
            self.pat_entities, score_threshold=80, num_articles=num_articles
        )
        self.entities = deepcopy(self.oa_sample_data)
        self.entities.update(self.pat_sample_data)
        self.next(self.filter_entities)

    @step
    def filter_entities(self):
        self.filtered_entities = filter_entities(
            self.sample_data, method="absolute", min_freq=10, max_freq=1_000_000
        )
        self.next(self.preprocess)

    @step
    def preprocess(self):
        self.filtered_entities = pipe(
            self.filtered_entities.values(), lambda oa: chain(*oa), set, list
        )
        self.next(self.embed)

    # @batch(
    #     queue="job-queue-GPU-nesta-metaflow",
    #     image="metaflow-pytorch",
    #     gpu=1,
    #     memory=60000,
    #     cpu=8,
    # )
    @pip(libraries={"sentence-transformers": "2.2.2"})
    @step
    def embed(self):
        from sentence_transformers import SentenceTransformer
        from torch import cuda

        if not cuda.is_available():
            raise EnvironmentError("CUDA is not available")

        self.model_name = MODEL_NAME
        encoder = SentenceTransformer(self.model_name)
        self.embeddings = encoder.encode(self.filtered_entities)
        self.next(self.save)

    @step
    def save(self):
        import pandas as pd

        self.embeddings_df = pd.DataFrame(self.embeddings)
        self.embeddings_df.to_parquet(f"s3://{BUCKET_NAME}/outputs/embeddings.parquet")
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    GetEmbeddings()
# %%
