# Entity Pre- and Post- processing Validation

This directory contains the scripts needed to validate the pre- and post- processing steps for entity enrichment.

## Generate Validation Sample

A sample of 100 documents was generated with their entities (with confidence thresholds of at least 50), the associated confidence scores and the spaCy NER label associated to them.

200 entities from 56 unique documents were labelled for whether:

- we would want to keep or remove them;
- if remove, the reason for removal (the four options were: `false_positive`, `other_reason`, `low_confidence_score`, and `bad_class`)
- if the spaCy label was correct or not

Overall, 75% of entities are entities we would want to keep while 25% of entities are entities we would want to remove.

## Confidence Threshold

- Confidence scores are in multiples of 10
- The average confidence score of bad entities is 61.2 while the median confidence score is 60
- The average confience score of good entities is 74.0 while the median confidence score is 70

| Confidence Threshold | % Entities Removed | % of 'Good' Entities |
| -------------------- | ------------------ | -------------------- |
| 60                   | 43.3               | 81.2                 |
| **70**               | **57.4**           | **88.8**             |
| 80                   | 65.0               | 87.3                 |
| 90                   | 88.7               | 91.0                 |

<img width="254" alt="good_conf_scores_dist" src="https://user-images.githubusercontent.com/46863334/207637066-8b972de9-b73c-4685-82d5-04946a5091ad.png">
<img width="254" alt="bad_conf_scores_dist" src="https://user-images.githubusercontent.com/46863334/207637090-11571c65-a831-44a4-a893-f445437efd05.png">

Based on the analysis, a confidence threshold score of **70** appears optimal for both entity quality and completeness.

## Entity Pre-Processing via classes

- Of all the entities in the validation sample, 56 had classes associated to them.
- Of the entities in the _labelled_ validation sample with classes associated to them, 16 had classes associated to them.
- There was 1 entity in the _labelled_ validation sample with classes associated to them labelled as 'bad'. The class looked appropriate and would be worth keeping.

Given the lack of classes associated to entities, it appears pre-processing **should not be included** in entity processing.

## Entity Post-Processing

- 34 of the 200 entities (or 17%) would be removed if we encorporated post-processing by removing entities that spaCy's pre-trained NER model predicts as a person, organisation, money or location.
- Of the 34 that we would want to remove, 76% of entities are incorrectly labelled.

<img width="254" alt="matrix" src="https://user-images.githubusercontent.com/46863334/207637144-fecb13c8-a0ea-40f2-918a-02353118e19e.png">

Based on the performance of the post-processing pipeline, it appears the pipeline **should not be included** in entity processing.
