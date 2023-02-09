# Embeddings

## Overview

The scripts in this folder generate semantic embeddings based on pre-trained large language models.We generate embeddings using the `allenai-specter` encoder, which is trained on a corpus of scientific articles and citations. Alternative encoders were not found to provide noticeable improvements in downstream tasks.

## Run it locally

To run the encoder locally, run `make_entity_embeddings.py` by executing the following command:

`python dap_aria_mapping/pipeline/embeddings/make_entity_embeddings.py --threshold=80 --num_articles=-1 --min_freq=10 --max_freq=1_000_000 --method=absolute --device=cuda --bucket=False`

We filter entities that have a confidence score below 80 (`--threshold`) and those that appear less than or more than a given number of times in the corpus. `--method=absolute` sets these lower and upper bounds using an absolute figure, while alternative `--method=percentile` takes values `0 < --min_freq < max_freq < 100`, where the bounds are set from percentiles given the distribution of entity observations across the corpus.

## Run it as a flow

Alternatively, one can run the embedding encoder as part of a metaflow, by executing `python dap_aria_mapping/pipeline/embeddings/flow.py run`.
