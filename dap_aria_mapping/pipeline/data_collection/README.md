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