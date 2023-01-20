# Semantic-based taxonomies

## Overview

This folder uses semantic embeddings to create a series of entity taxonomies. It leverages embeddings from the tags identified in articles extracted from OpenAlex or Google Patents, and uses them to cluster entities into groups. This taxonomy is assumed hierarchical.

Multiple options are explored to create the taxonomy, including:

- **Strict hierarchical clustering**: Iteratively cluster the entity embeddings using Agglomerative Clustering and KMeans. At each level, clustering is performed on the subsets that were created at the previous level. This allows for strict hierarchical entity breakdowns.
- **Strict hierarchical clustering with varying number of clusters**: Iteratively cluster the entity embeddings using KMeans. At each level, clustering is performed on the subsets that were created at the previous level. The number of clusters is proportional to the parent cluster size.
- **Fuzzy hierarchical clustering**: Cluster the entity embeddings using an array of methods, including KMeans, Agglomerative Clustering, DBSCAN, and HDBSCAN. Several levels of resolution are used, and the taxonomy is built by concatenating these. This allows for non-strict hierarchical entity breakdowns.
- **Hierarchical clustering using a dendrogram climb algorithm**: Reconstruct the dendrogram of the Agglomerative Clustering, and use it to create the taxonomy.
- **Hierarchical clustering using the centroids of parent-level clusters**: Cluster the entity embeddings using KMeans, and use the centroids of the clusters as nodes in subsequent levels of the taxonomy.
- **Meta clustering using the co-occurrence of terms in the set of clustering results**: Use outputs fromall previous clustering methods to create a tag co-occurrence matrix. Apply community detection algorithms to this matrix to create the taxonomy.

The utils for the different approaches can be found in `utils.semantics.py` and include all necessary functions to create the taxonomy, including the class ClusteringRoutine that performs any of the clustering methods described above.

## Single Clustering routine

The `make_taxonomy.py` file contains the basic routine and function calls necessary to build a single taxonomy. To run an example, execute the following:

`python dap_aria_mapping/pipeline/semantic_taxonomy/make_taxonomy.py --cluster_method=centroids --plot=True --production=True`

Note that config either corresponds to a clustering routine defined in the basic config file, or a standalone yaml file with the necessary parameters. These configurations accept lists within cluster parameters - which outputs the product of all possible configurations within a single cluster level -, and lists of cluster parameters, which are interpreted as different levels of the algorithm. See the class docstrings in `utils.semantics.py` for additional details.

![1673627635364](image/README/1673627635364.png)

The outputs are twofold:

- A list of dictionaries with per-level label assignments, stored in `s3://aria-mapping/outputs/semantic_taxonomy/raw_outputs/`
- A dataframe of cluster assignments, with rows corresponding to entities and columns corresponding to different levels of the taxonomy. These are stored in `s3://aria-mapping/outputs/semantic_taxonomy/assignments/`

## Manifold Clustering routine

An alternative to selecting one of several clustering approaches is to employ all of them by running a community detection algorithm on the co-occurrences of entities within clusters across all approaches.

The file `make_taxonomies.py` generates taxonomies using all methods described above, and exports additional outputs over the single routine as it also produces matrices of co-occurrences for each method. In addition, it combines all clustering algorithms, and generates a co-occurrence matrix for the frequencies of co-occurrences among all approaches, which we label `meta_cluster`. These are stored in `s3://aria-mapping/outputs/semantic_taxonomy/cooccurrences/`.

## Make a Graph object

In order to use the `meta_cluster` in a community detection algorithm, we use the matrix of co-occurrences as an adjacency matrix to construct a graph object `nx.Graph`. To do so, execute:

`python dap_aria_mapping/pipeline/semantic_taxonomy/make_graph.py --cluster_object=meta_cluster --bucket_name==aria-mapping`

This outputs a graph object which can be used in the existing pipeline used to construct a taxonomy through entity co-occurrences. The difference is that the meaning of the edge weights between terms represent their co-occurrences on semantic-based clustering algorithms, and not only their joint presence in OpenAlex and Google Patents data.

### Sample dataset

We illustrate an example of the manifold clustering routine using a sample of 3,000 articles. Tags in these entities are filtered by selecting those with a confidence threshold higher than 80 and a minimum frequency above the 60th percentile of other entities.

![1671699887328](image/README/1671699887328.png)

### Strictly hierarchical clustering

The following clustering routine iteratively clusters the entity embeddings using Agglomerative Clustering and KMeans. At each level, clustering is performed on the subsets that were created at the previous level.

![1671699928260](image/README/1671699928260.png)

##### Cluster histograms

###### Kmeans

![1671700177608](image/README/1671700177608.png)

###### Agglomerative Clustering![1671700296566](image/README/1671700296566.png)

### Strict hierarchical clustering with imbalanced nested clusters

The following clustering routine iteratively clusters the entity embeddings using KMeans. At each level, clustering is performed on the subsets that were created at the previous level. The number of clusters at each level is allowed to vary, being determined by the size of the parent cluster.

![1671699987259](image/README/1671699987259.png)

##### Barplots![1671700328908](image/README/1671700328908.png)

### Fuzzy hierarchical clustering

The following approach iteratively clusters the entity embeddings using any sklearn method that supports the `predict_proba` method. No notion of level exists through this approach: more fine-grained clusterings are agnostic about the parent cluster output. Including several lists of parameter values will produce outputs for the Cartesian product of all parameter values within a clustering method.

![1671700389849](image/README/1671700389849.png)

DBSCAN and HDBSCAN are unable to output meaningful clusters given a set of parameters (DBSCAN: eps 0.05 & 0.1, min_samples 4 & 8; HDBSCAN: min_cluster_size 4 & 8, min_samples 4 & 8), where each column in the subplot corresponds to one pairing of parameters.

### Using dendrograms from Agglomerative Clustering (enforces hierarchy)

This approach uses a single run of any sklearn clustering method that supports a children* attribute. The children* attribute is used to recreate th dendrogram that produced the clustering, which is then used to create the taxonomy. The climbing algorithm advances one union of subtrees at a time. The number of levels is determined by the `dendrogram_levels` parameter.

![1671700575409](image/README/1671700575409.png)

##### Barplots

![1671700601164](image/README/1671700601164.png)

### Centroids of Kmeans clustering as children nodes for further clustering (à la [job skill taxonomy](https://github.com/nestauk/skills-taxonomy-v2/tree/dev/skills_taxonomy_v2/pipeline/skills_taxonomy))

This approach uses any number of nested KMeans clustering runs. After a given level, the centroids of the previous level are used as the new data points for the next level.![1671700624717](image/README/1671700624717.png)![1671700732629](image/README/1671700732629.png)

### Silhouette Scores

We compute silhouette scores to inform the relative performance of each method at their respective level. Note that these are not fine tuned, and as such levels are not comparable across methods.

| Method - level             | Silhouette Score |
| -------------------------- | ---------------- |
| agglomerative_dendrogram_0 | 0.111172         |
| kmeans_centroid_0          | 0.070699         |
| kmeans_strict_imb_0        | 0.058071         |
| kmeans_strict_0            | 0.057617         |
| agglomerative_dendrogram_2 | 0.037812         |
| kmeans_centroid_1          | 0.035893         |
| kmeans_centroid_3          | 0.035776         |
| agglom_strict_0            | 0.034922         |
| agglomerative_dendrogram_1 | 0.034922         |
| kmeans_fuzzy_3             | 0.026868         |
| agglomerative_dendrogram_3 | 0.024250         |
| agglom_strict_1            | 0.023158         |
| agglom_fuzzy_0             | 0.020661         |
| kmeans_fuzzy_2             | 0.019473         |
| kmeans_fuzzy_0             | 0.018972         |
| kmeans_strict_1            | 0.018489         |
| kmeans_strict_2            | 0.017510         |
| kmeans_fuzzy_1             | 0.011748         |
| agglom_fuzzy_3             | 0.007710         |
| agglom_fuzzy_2             | 0.001148         |
| kmeans_strict_imb_1        | -0.000620        |
| kmeans_strict_imb_2        | -0.003070        |
| agglom_strict_2            | -0.007311        |
| agglomerative_dendrogram_4 | -0.010194        |
| agglom_fuzzy_1             | -0.015053        |
| kmeans_centroid_2          | -0.021420        |
| agglomerative_dendrogram_5 | -0.025106        |

### Meta Clustering

Following the approach of Juan in the [AFS repository](https://github.com/nestauk/afs_neighbourhood_analysis/tree/1b1f1b1dabbebd07e5c85c72d7401107173bf863/afs_neighbourhood_analysis/analysis), we combine the clustering methods to produce a matrix of entity co-occurrences. The objective is to apply community detection algorithms on this.