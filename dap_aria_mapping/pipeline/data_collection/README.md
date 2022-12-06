# Data Collection

## Overview

Here you will find files related to the collection of data for the DAP ARIA Mapping project.

## OpenAlex

OpenAlex is a tool that allows you to search for publications on the web and download them. To run the collection, execute the following:

```python dap_aria_mapping/pipeline/data_collection/openalex.py --package-suffixes=.txt --datastore=s3 run --production=True```

This will collect all UK publications from the past 16 years and save each year as a file in the S3 bucket. If the `--production=true` flag is not set, the script will run in test mode and only collect publications from a single year, for debugging and testing purposes.

## Processed OpenAlex

This flow joins all the outputs of the previous flow and saves them as a single file in the S3 bucket. To run the collection, execute the following:

```python dap_aria_mapping/pipeline/data_collection/processed_openalex.py --datastore=s3 run --production=True```

This will collect all UK publications from the past 16 years and save them as a single file in the S3 bucket. There is also a level of normalisation; works, concepts, abstracts, authorships and citations are saved separately. If the `--production=true` flag is not set, the script will run in test mode and only collect publications from a single year, for debugging and testing purposes.

## Getting Up and Running With Batch on Metaflow

If you haven't used batch processing with Metaflow before and want to run any of the flows that make use of batch (e.g. `openalex.py`), you'll need to ensure a few things are set up first:

1. Your metaflow config file needs be setup with the correct parameters. You can find your config file by executing `metaflow configure show`. If you don't have parameters such as `METAFLOW_ECS_S3_ACCESS_IAM_ROLE` and `METAFLOW_ECS_FARGATE_EXECUTION_ROLE`, contact the DE team.
2. If your laptop username contains a `.` (e.g. if you run `whoami` from the command line and it returns `jack.vines` rather than `jackvines`), you'll need to change your username to remove the `.`. This is because the AWS Batch job will fail to run if your username contains a `.`. To fix this, add `export METAFLOW_USER=<your name without the period>` to a `.env` file at the root of the project. Then, [one time only] run `direnv reload` to trigger the reloading of .envrc (which itself sources .env)