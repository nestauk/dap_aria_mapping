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

☁️ The `openalex.py` and `processed_openalex.py` pipelines are set up to run in batch mode. This means that they will run on AWS Batch, and require some configuration. Please [see below](#getting-up-and-running-with-batch-on-metaflow) for further details if you want to run these flows.

We have also included pytest unit tests for relevant functions in the tests directory.

Output data from these pipelines is stored on S3 in the `inputs/data_collection` directory. Getters for these datasets have been added to the getters section of the repository.

### Additional Raw Data

#### Foward citations for measuring disruption

To enable the calculation of the consolidation-disruption (CD) index for OpenAlex works, a separate pipeline must be run. This collects the work IDs and citations of all works that cite the 'focus works' collected by `dap_aria_mapping/pipeline/data_collection/openalex.py`. It is a separate flow as it generates a large number of API calls.

To run this collection, execute the following:

`python dap_aria_mapping/pipeline/data_collection/openalex_forward_citations.py --datastore=s3 run --production=True`

Output data from these pipelines is stored on S3 at `inputs/data_collection/openalex/openalex_forward_citations(_focus_papers_only)_<year>.json`.

⚠️ The default parameters collect citations for works between 2007 and 2022 that have a minimum of 3 citations. This will generate a large volume of API calls and may result in rate limiting. You may want to use the `min_year` and `max_year` parameters to run the pipeline for individual years. The `min_citations` parameter can also be varied to exclude the collection of works that cite focus works with fewer than a particular number of citations.

☁️ This pipeline is set up to run in batch mode. This means that it will run on AWS Batch, and requires some configuration. Please [see below](#getting-up-and-running-with-batch-on-metaflow) for further details if you want to run these flows.

## Getting Up and Running With Batch on Metaflow

If you haven't used batch processing with Metaflow before and want to run any of the flows that make use of batch (e.g. `openalex.py`), you'll need to ensure a few things are set up first:

1. Your metaflow config file needs be setup with the correct parameters. You can find your config file by executing `metaflow configure show`. If you don't have parameters such as `METAFLOW_ECS_S3_ACCESS_IAM_ROLE` and `METAFLOW_ECS_FARGATE_EXECUTION_ROLE`, contact the DE team.
2. If your laptop username contains a `.` (e.g. if you run `whoami` from the command line and it returns `jack.vines` rather than `jackvines`), you'll need to change your username to remove the `.`. This is because the AWS Batch job will fail to run if your username contains a `.`. To fix this, add `export METAFLOW_USER=<your name without the period>` to a `.env` file at the root of the project. Then, [one time only] run `source .env` to trigger reloading of the variable.
