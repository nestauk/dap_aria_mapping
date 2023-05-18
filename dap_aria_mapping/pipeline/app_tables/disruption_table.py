from dap_aria_mapping.getters.openalex import (
    get_openalex_topics,
    get_openalex_cd_scores,
    get_openalex_authorships,
)
from dap_aria_mapping.utils.app_data_utils import (
    expand_topic_col,
    add_area_domain_chatgpt_names,
)
from dap_aria_mapping.getters.novelty import get_openalex_novelty_scores
import polars as pl
import pandas as pd
from dap_aria_mapping import logger, BUCKET_NAME
import io
import boto3
from toolz import pipe

logger.info("Loading and aggregating publication cd scores")
years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
cd_scores = pl.DataFrame(data=None)
for year in years:
    df = pl.DataFrame(
        pd.DataFrame.from_dict(get_openalex_cd_scores(year), orient="index")
        .T.unstack()
        .dropna()
        .reset_index(drop=True, level=1)
        .to_frame()
        .reset_index()
    ).with_columns(pl.lit(year).alias("year"))
    cd_scores = pl.concat([cd_scores, df])
cd_scores.columns = ["id", "cd_score", "year"]
logger.info("Have c-d scores for {} publications".format(len(cd_scores)))

pubs_with_topics_df = (
    pl.DataFrame(
        pipe(
            get_openalex_topics(tax="cooccur", level=3),
            lambda di: [(k, v) for k, v in di.items() if len(v) > 0],
        ),
        schema=["id", "topics"],
    )
    .lazy()
    .explode("topics")
)

pubs_with_topics_df = pubs_with_topics_df.collect()
