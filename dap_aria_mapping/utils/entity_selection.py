import numpy as np
import pandas as pd
import re, sklearn
import matplotlib.axes as mpl_axes
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from typing import (
    Union,
    Dict,
    DefaultDict,
    List,
    Any,
    Set,
    Type,
    Literal,
    Sequence,
    Tuple,
    Generator,
)
from itertools import product
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import chain, count
import altair as alt

alt.data_transformers.disable_max_rows()


def get_sample(
    entities: Dict[str, Sequence[Tuple[str, int]]],
    score_threshold: int = 80,
    num_articles: int = -1,
) -> Dict[str, Sequence[str]]:
    """Creates a sample of entities from the entities dictionary,
    where each article contains a list of tuples of the form (entity, score).
    The sample size is determined by `num_articles`, and entities with
    a confidence score below the threshold are removed.
    Args:
        entities (Dict[str, Sequence[Tuple[str, int]]]): A dictionary of entities,
            where each key is an article and each value is a list of tuples of the
            form (entity, score).
        score_threshold (int, optional): The minimum confidence score for an entity.
            If not provided, defaults to 80.
        num_articles (int, optional): The number of articles to sample. If -1,
            all articles are sampled. If not provided, defaults to -1.
    Returns:
        Dict[str, Sequence[str]]: A dictionary of entities, where each key is
            an article and each value is a list of entities.
    """
    entities = {k: v for k, v in entities.items() if len(v) > 0}
    if num_articles == -1:
        num_articles = len(entities.values())
    draws = num_articles / len(entities.values()) >= np.random.uniform(
        0, 1, len(entities.values())
    )
    sample = {k: v for i, (k, v) in enumerate(entities.items()) if draws[i]}
    return {
        k: [e["entity"] for e in v if e["confidence"] >= score_threshold]
        for k, v in sample.items()
    }


def filter_entities(
    entities: Dict[str, Sequence[str]],
    method: Literal["percentile", "absolute"] = "percentile",
    min_freq: int = 0,
    max_freq: int = 100,
) -> Dict[str, Sequence[str]]:
    """Filters entities from the entities dictionary, where each article contains
    a list of entities. The filtering is done by frequency, either as a percentile
    or as an absolute value.
    Args:
        entities (Dict[str, Sequence[str]]): A dictionary of entities, where each
            key is an article and each value is a list of entities.
        method (Literal[percentile, absolute], optional): The method of filtering.
            Defaults to "percentile".
        min_freq (int, optional): The minimum frequency for an entity. If `method`
            is "percentile", this is the minimum percentile. If `method` is "absolute",
            this is the minimum absolute frequency. Defaults to 0.
        max_freq (int, optional): The maximum frequency for an entity. If `method`
            is "percentile", this is the maximum percentile. If `method` is "absolute",
            this is the maximum absolute frequency. Defaults to 100.
    Returns:
        Dict[str, Sequence[str]]: A dictionary of entities, where each key is
            an article and each value is a list of entities.
    """
    assert min_freq < max_freq, "min_freq must be less than max_freq"
    assert method in [
        "percentile",
        "absolute",
    ], "method must be in ['percentile', 'absolute']"
    frequencies = Counter(chain(*entities.values()))
    if method == "percentile":
        min_freq, max_freq = (
            np.percentile(list(frequencies.values()), min_freq),
            np.percentile(list(frequencies.values()), max_freq),
        )
    return {
        k: [e for e in v if min_freq <= frequencies[e] <= max_freq]
        for k, v in entities.items()
    }
