# Simulations

This folder simulates the production of taxonomies under two stress conditions:

- **noise**: add entities discarded during pre-processing to test the ability of any taxonomy to still capture relevant data / topic structures and ignore isolated noise entities.
- **sample**: run the taxonomy on sub-sets of the entity data to test the taxonomy model's robustness to sample size and topic convergence.

The folders `noise` and `sample` include scripts for the building of taxonomies under their respective stress situations. The `noise` scripts automatically run ten simulations in a log-scale of noise, which is measured as the percentage of noisy entities to the post-processed set of entities. It log-ranges from 5% to 51%, and an example (for the cooccurrence taxonomy) can be run by executing the following in the command window:

`python dap_aria_mapping/pipeline/taxonomy_validation/simulations/noise/make_network_simulations.py --n_simulations 1 -l --seed 42`

Conversely, sample simulations require a list of sample fractions, ie.

`python dap_aria_mapping/pipeline/taxonomy_validation/simulations/sample/make_network_simulations.py --n_simulations 1 -l --seed 42 --sample_sizes 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90`

Note that several simulations can be produced at each configuration, although analysis is conducted assuming a single simulation for noise or sample configuration is run.

## Topic names

Simulated taxonomies are labelled via their most frequent entities, which are changing across noise and sample configurations. In order to produce topic entity names for all simulations, an example command line is:

`python dap_aria_mapping/pipeline/taxonomy_validation/simulations/make_name_assignments.py --taxonomy community --simulation noise sample --n_top 5 --n_articles 100000 --save`

This script generates the entity counts after parsing 100,000 articles (roughly 3% of our dataset), and assigns them to topics proportional to their values.

## Sensitivity data

The purpose of these simulations is to conduct sensitivity analysis, whereby different stress conditions of sample sizes or noisy entities should affect the relative performance of the candidate taxonomies. In order to conduct this analysis, we produce the following metrics:

- **relative topic sizes:** used to test whether the introduction of noise or sample selection creates larger or smaller principal components, as well as the distribution of sizes.
- **distances to topic centroids**: used to evaluate whether pre-processed entities end up in more disperse average topics when reducing the data size or adding noise.
- **pairwise_entities**: given a pre-specified set of entities that field experts consider should be included within a given topic, we test the frequency with which this is true under different stress conditions.

In order to produce these datasets, an example command line is:

`python dap_aria_mapping/pipeline/taxonomy_validation/simulations/make_sensitivity_data.py --taxonomy community centroids imbalanced --simulation sample noise --save`
