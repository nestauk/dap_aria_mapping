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
