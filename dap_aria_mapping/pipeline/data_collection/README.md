# Data Collection

## Overview

This directory contains code for collecting and processing data from OpenAlex and Google Patents for UK academic publications and patents.

## Raw Data Collection

The `openalex.py` file contains the pipeline for collecting raw publication data from OpenAlex. The `patents.py` file contains the pipeline for collecting raw patent data from Google Patents.

To run the raw collection, execute the following:

`python dap_aria_mapping/pipeline/data_collection/openalex.py --package-suffixes=.txt --datastore=s3 run --production=True`

`python dap_aria_mapping/pipeline/data_collection/patents.py run --production=True`

The `processed_openalex.py` file contains the pipeline for processing the raw publication data and the `processed_patents.py` file contains the pipeline for processing the raw patent data.

## Raw Data Processing

To run the processed collection, execute the following:

`python dap_aria_mapping/pipeline/data_collection/processed_openalex.py --datastore=s3 run --production=True`

`python dap_aria_mapping/pipeline/data_collection/processed_patents.py run`

Note: The `openalex.py` and `processed_openalex.py` pipelines are set up to run in batch mode. This means that they will run on AWS Batch, and require some configuration. Please see below for further details if you want to run these flows.

We have also included pytest unit tests for relevant functions in the tests directory.

Output data from these pipelines is stored on S3 in the `inputs/data_collection` directory. Getters for these datasets have been added to the getters section of the repository.

### Additional Raw Data

#### Foward citations for measuring disruption

#### Patents

We also collect citation information for patents to calculate the consolidation-distruption (CD) index.

We define 'focus patents' as the patent ids collected by `dap_aria_mapping/pipeline/data_collection/openalex.py.`.

To collect the patent forward citations, backward citations and backward citation metadata, run the following flow:

`python dap_aria_mapping/pipeline/data_collection/paents_citations.py run --production=False`

If you run the flow in production (i.e. `production=True`), it will cost money to do so.

This flow outputs three files stored on S3 at `inputs/data_collection/patents/`:

1. **`patents_forward_citations.json`**: A json where the key is a focal patent id and the value is a list of patent ids that cite the focal patent id.
2. **`patents_backward_citations.json`**: A json where the key is a focal patent id and the value is a list of patent ids that the focal patent id cites.
3. **`patents_citations_metadata.parquet`**: A parquet file that stores additional backward citation metadata including citation category, citation type, free-text citation in non-patent literature and the citation filing date.

i.e.

Please refer to the [patents data dictionary](https://docs.google.com/spreadsheets/d/1LtfjECVI5pqqwE7oMw1JbwFcUWhUoHgJH_mJ0flw9Fw/edit#gid=1878548964) for additional information on the collection information.

## Getting Up and Running With Batch on Metaflow

If you haven't used batch processing with Metaflow before and want to run any of the flows that make use of batch (e.g. `openalex.py`), you'll need to ensure a few things are set up first:

1. Your metaflow config file needs be setup with the correct parameters. You can find your config file by executing `metaflow configure show`. If you don't have parameters such as `METAFLOW_ECS_S3_ACCESS_IAM_ROLE` and `METAFLOW_ECS_FARGATE_EXECUTION_ROLE`, contact the DE team.
2. If your laptop username contains a `.` (e.g. if you run `whoami` from the command line and it returns `jack.vines` rather than `jackvines`), you'll need to change your username to remove the `.`. This is because the AWS Batch job will fail to run if your username contains a `.`. To fix this, add `export METAFLOW_USER=<your name without the period>` to a `.env` file at the root of the project. Then, [one time only] run `source .env` to trigger reloading of the variable.
