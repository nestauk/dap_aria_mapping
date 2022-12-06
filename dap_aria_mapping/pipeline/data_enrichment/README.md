# Data Enrichment

## Overview

This directory contains code for enriching data from OpenAlex and Google Patents for UK academic publications and patents.

## Entity Post-processing

The `processed_entities.py` file contains the pipeline for post-processing extracted DBpedia entities from OpenAlex and Patents. It filters out entities that are predicted as: people, organisations, locations and money using spaCy's pretrained NER model. It also filters entities with a confidence score below 60. Finally, it also strips the confidence score associated to a given entity.

To run data enrichement, execute the following:

`python dap_aria_mapping/pipeline/data_enrichment/processed_entities.py run`

We have also included pytest unit tests for relevant functions in the tests directory.

Output data from this pipeline is stored on S3 in the inputs/data_enrichment/extracted_entities directory. Getters for these datasets have been added to the getters section of the repository.
