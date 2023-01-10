from dap_aria_mapping import BUCKET_NAME, PROJECT_DIR, logger
from toolz import pipe
from itertools import chain
from functools import partial
import pickle, argparse
import boto3
from copy import deepcopy

from dap_aria_mapping.pipeline.semantic_taxonomy.utils import (
    get_sample,
    filter_entities,
)
from dap_aria_mapping.getters.openalex import get_openalex_entities
from dap_aria_mapping.getters.patents import get_patent_entities
from sentence_transformers import SentenceTransformer
from torch import cuda

MODEL_NAME = "sentence-transformers/allenai-specter"

parser = argparse.ArgumentParser(
    description="Create embeddings for OpenAlex entities",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--threshold", default=80, type=int, help="Threshold for OpenAlex scores"
)
parser.add_argument(
    "--num_articles", default=3_000, type=int, help="Number of articles to sample"
)
parser.add_argument(
    "--min_freq", default=60, type=int, help="Minimum frequency for entities"
)
parser.add_argument(
    "--max_freq", default=100, type=int, help="Maximum frequency for entities"
)
parser.add_argument(
    "--method",
    default="percentile",
    help="Method for filtering entities, either 'percentile' or 'absolute'",
)
parser.add_argument("--model", default=MODEL_NAME, help="Name for the model to use")
parser.add_argument(
    "--device", default="cuda" if cuda.is_available() else "cpu", help="Device to use"
)

parser.add_argument(
    "--bucket",
    default=False,
    help="Bucket to use for saving the embeddings. If False, saves locally",
)
args = parser.parse_args()

logger.info("Getting OpenAlex entities")
openalex_entities = pipe(
    get_openalex_entities(),
    partial(get_sample, score_threshold=args.threshold, num_articles=args.num_articles),
)

logger.info("Getting patent entities")
patent_entities = pipe(
    get_patent_entities(),
    partial(get_sample, score_threshold=args.threshold, num_articles=args.num_articles),
)

entities = deepcopy(openalex_entities)
entities.update(patent_entities)
entities = pipe(
    entities,
    partial(
        filter_entities,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        method=args.method,
    ),
    lambda dict: dict.values(),
    lambda entities: chain(*entities),
    set,
    list,
)


logger.info("Creating embeddings")
# embeddings
encoder = SentenceTransformer(MODEL_NAME, device=args.device)
embeddings = pipe(entities, encoder.encode, lambda embeds: dict(zip(entities, embeds)))

logger.info("Saving embeddings")
if args.bucket:
    s3 = boto3.client("s3")
    s3.put_object(
        Body=pickle.dumps(embeddings),
        Bucket=BUCKET_NAME,
        Key="outputs/embeddings/embeddings.pkl",
    )
else:
    with open(f"{PROJECT_DIR}/outputs/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
