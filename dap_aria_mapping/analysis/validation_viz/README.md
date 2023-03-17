# Exploration of taxonomies and validation metrics

Th streamlit app collates a series of performance and validation metrics & tests for aeach of three taxonomy candidates, namely **networks of cooccurrences**, **imbalanced hierarchical KMeans clustering**, and **centroids-based hierarchical KMeans clustering**. The series of performance and simulation metrics include:

- **Distribution metrics**:
  - histograms with topic counts per level and taxonomy.
  - frequency of topic splits per level and taxonomy.
  - KL-divergence vis-à-vis uniform distributions.
  - $\chi^{2}$ test statistic of entity distributions per topic (vs. uniform).
- **Content metrics:**
  - frequencies with which manually identified same-topic entities are in the same topic.
  - frequencies with which manually identified same-subtopic entities are in the same subtopic.
- **Simulation metrics** (either simulating the addition of noisy entities, or reducing sample size):
  - changes in topic composition and size.
  - average distance to centroid of non-noisy entities.
  - frequencies of manually identified same-topic and same-subtopic entities being in the same cluster.
- **Journal metrics**:
  - histograms with counts of relevant topics stemming from each main journal at a given taxonomy level.
  - barplots with frequencies of topic assignment given entities at a given taxonomy level (for relevant journals).

To run the streamlit app, install the library and then run the following command line:

`streamlit run dap_aria_mapping/analysis/validation_viz/taxonomy_validation.py`

# Exploration of chatGPT names

The `pipeline/taxonomy_validation/make_topic_name_assignments.py` routine includes chatgpt bots that request names for topics given a list of entities and their counts. Additionally, the query includes requests for chatgpt to discard those entities it considers noisy given its topic name answer.

The outputs from the labelling are lists of four-item tuples, corresponding to topic ID, topic label, confidence score, and discarded entities, ie.

`[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioshka', 'Madrid'])]`

All these results are presented in the dataframes as part of the streamlit app.
