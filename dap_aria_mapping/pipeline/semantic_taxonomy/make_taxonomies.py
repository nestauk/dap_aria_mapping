# %% [markdown]
# ## Exploration of semantic-based taxonomies
# The following notebook explores the use of semantic embeddings to create taxonomies of entities, with the goal of creating an ontology of the ARIA dataset. It leverages embeddings from the tags identified in articles extracted from OpenAlex, and uses them to cluster entities into groups. This taxonomy is assumed hierarchical.
#
# Multiple options are explored to create the taxonomy, including:
# - **Strict hierarchical clustering**: Iteratively cluster the entity embeddings using Agglomerative Clustering and KMeans. At each level, clustering is performed on the subsets that were created at the previous level. This allows for strict hierarchical entity breakdowns.
# - **Strict hierarchical clustering with varying number of clusters**: Iteratively cluster the entity embeddings using KMeans. At each level, clustering is performed on the subsets that were created at the previous level. The number of clusters is proportional to the parent cluster size.
# - **Fuzzy hierarchical clustering**: Cluster the entity embeddings using an array of methods, including KMeans, Agglomerative Clustering, DBSCAN, OPTICS, and HDBSCAN. Several levels of resolution are used, and the taxonomy is built by concatenating these. This allows for non-strict hierarchical entity breakdowns.
# - **Hierarchical clustering using a dendrogram climb algorithm**: Reconstruct the dendrogram of the Agglomerative Clustering, and use it to create the taxonomy.
# - **Hierarchical clustering using the centroids of parent-level clusters**: Cluster the entity embeddings using KMeans, and use the centroids of the clusters as nodes in subsequent levels of the taxonomy.
# - **Meta clustering using the co-occurrence of terms in the set of clustering results**: Use outputs from all previous clustering methods to create a tag co-occurrence matrix. Apply community detection algorithms to this matrix to create the taxonomy.
#
# The utils for this script include all necessary functions to create the taxonomy, including class ClusteringRoutine that performs any of the clustering methods described above. The function run_clustering_generators is used to run all clustering methods and return the results in a dictionary. The function make_dataframe is used to create a dataframe with the results of the clustering methods. The function make_plots is used to create a series of plots to visualize the results of the clustering methods. The function make_cooccurrences is used to create a co-occurrence matrix of the clustering results. The function make_subplot_embeddings is used to create a series of subplots with the embeddings of the entities in the taxonomy.

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import boto3, pickle
from IPython.display import display
import numpy as np
import pandas as pd
import umap.umap_ as umap
import networkx as nx
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from hdbscan import HDBSCAN
from collections import defaultdict
from dap_aria_mapping import PROJECT_DIR, BUCKET_NAME, logger, config
from dap_aria_mapping.utils.semantics import (
    make_dataframe,
    make_cooccurrences,
    run_clustering_generators,
)

np.random.seed(42)
cluster_configs, df_configs = [
    config["SEMANTIC_TAXONOMY"][key] for key in ["CLUSTER_CONFIGS", "DF_CONFIGS"]
]


if __name__ == "__main__":
    logger.info("Downloading embeddings from S3")
    s3 = boto3.client("s3")
    embeddings_object = s3.get_object(
        Bucket=BUCKET_NAME, Key="outputs/embeddings/embeddings.pkl"
    )
    embeddings = pickle.loads(embeddings_object["Body"].read())
    embeddings = pd.DataFrame.from_dict(embeddings).T

    embeddings_2d = umap.UMAP(
        n_neighbors=5, min_dist=0.05, n_components=2
    ).fit_transform(embeddings)

    CLUSTER_CONFIGS = {
        "strict": [
            [KMeans, cluster_configs["strict_kmeans"]],
            [AgglomerativeClustering, cluster_configs["strict_agglom"]],
        ],
        "strict_imb": [[KMeans, cluster_configs["strict_kmeans_imb"]]],
        "fuzzy": [
            [KMeans, cluster_configs["fuzzy_kmeans"]],
            [AgglomerativeClustering, cluster_configs["fuzzy_agglom"]],
            [DBSCAN, cluster_configs["fuzzy_dbscan"]],
            [HDBSCAN, cluster_configs["fuzzy_hdbscan"]],
        ],
        "dendrogram": [[AgglomerativeClustering, cluster_configs["dendrogram"]]],
        "centroids": [[KMeans, cluster_configs["centroids_test"]]],
    }

    logger.info("Running clustering generators - Strict hierarchical clustering")
    # run clustering generators
    cluster_outputs_s, _ = run_clustering_generators(
        CLUSTER_CONFIGS["strict"], embeddings
    )
    logger.info(
        "Running clustering generators - Strict hierarchical clustering with imbalanced nested clusters"
    )
    # Strict hierarchical clustering with imbalanced nested clusters
    cluster_outputs_simb, _ = run_clustering_generators(
        CLUSTER_CONFIGS["strict_imb"], embeddings, imbalanced=True
    )
    logger.info("Running clustering generators - Fuzzy hierarchical clustering")
    # Fuzzy hierarchical clustering
    cluster_outputs_f_, _ = run_clustering_generators(
        CLUSTER_CONFIGS["fuzzy"], embeddings
    )
    logger.info("Running clustering generators - Dendrogram")
    # Using dendrograms from Agglomerative Clustering (enforces hierarchy)
    cluster_outputs_d, _ = run_clustering_generators(
        CLUSTER_CONFIGS["dendrogram"], embeddings, dendrogram_levels=200
    )
    logger.info(
        "Running clustering generators - Centroids of Kmeans clustering as children nodes for further clustering"
    )
    # Centroids of Kmeans clustering as children nodes for further clustering (à la [job skill taxonomy](https://github.com/nestauk/skills-taxonomy-v2/tree/dev/skills_taxonomy_v2/pipeline/skills_taxonomy))
    cluster_outputs_c, _ = run_clustering_generators(
        CLUSTER_CONFIGS["centroids"], embeddings, embeddings_2d=embeddings_2d
    )
    # Necessary changes to create comparable outputs from generators
    # Centroids (reverse order)
    for output_dict in cluster_outputs_c:
        for k, v in output_dict["labels"].items():
            output_dict["labels"][k] = v[::-1]
        output_dict["silhouette"] = output_dict["silhouette"][::-1]

    # Cluster outputs - fuzzy
    cluster_outputs_f = []
    for group in ["sklearn.cluster._kmeans", "sklearn.cluster._agglomerative"]:
        dict_group = {
            "labels": defaultdict(list),
            "model": [],
            "silhouette": [],
            "centroid_params": None,
        }

        cluster_op = [
            x for x in cluster_outputs_f_ if x["model"][0].__module__ == group
        ]
        for clust in cluster_op:
            for k, v in clust["labels"].items():
                dict_group["labels"][k].append(v[0])
            dict_group["model"].append(
                "_".join(
                    [
                        clust["model"][0].__module__.replace(".", ""),
                        str(clust["model"][0].get_params()["n_clusters"]),
                    ]
                )
            )
            dict_group["silhouette"].append(clust["silhouette"][0])
        cluster_outputs_f.append(dict_group)
    logger.info("Running clustering generators - Done")

    logger.info("Running dictionary & table exports")
    # Relevant dictionary outputs
    outputs_ls = [
        cluster_outputs_s[3],
        cluster_outputs_s[7],
        cluster_outputs_simb[-1],
        cluster_outputs_f[0],
        cluster_outputs_f[1],
        cluster_outputs_d,
        cluster_outputs_c[-1],
    ]

    # Create list of output dataframes
    outputs_df = []
    for dict_, df_config in zip(outputs_ls, df_configs):
        name_export = df_config["name"]
        s3.put_object(
            Body=pickle.dumps(dict_),
            Bucket=BUCKET_NAME,
            Key=f"outputs/semantic_taxonomy/raw_outputs/{name_export}.pkl",
        )

        df_ = make_dataframe(dict_, df_config["label"], df_config["cumulative"])
        if df_config["name"] == "agglom_dendrogram":  # avoid using all dgram levels
            depth_dendrogram = df_.shape[1]
            levels = [int(depth_dendrogram * i) for i in [0.1, 0.3, 0.5, 0.7, 0.9]]
            df_ = df_.iloc[:, levels]
        outputs_df.append(df_)

    # Silhouette Scores
    results = {
        "_".join([k["name"], str(id)]): e
        for v, k in list(zip(outputs_ls, df_configs))
        for id, e in enumerate(v["silhouette"])
    }

    silhouette_df = pd.DataFrame(results, index=["silhouette"]).T.sort_values(
        "silhouette", ascending=False
    )

    silhouette_df.to_parquet(
        f"s3://aria-mapping/outputs/semantic_taxonomy/silhouette_scores.parquet"
    )

    # Meta Clustering
    meta_cluster_df = pd.concat(outputs_df, axis=1)

    # Exports co-occurrences and tables of cluster assignments
    labels = [x["name"] for x in df_configs]
    for label, df in zip(labels + ["meta_cluster"], outputs_df + [meta_cluster_df]):
        df_ = df.reset_index().rename(columns={"index": "tag"})
        cooccur_dict = make_cooccurrences(df_)
        cooccur_df = pd.DataFrame(cooccur_dict, index=df_["tag"], columns=df_["tag"])

        df_.to_parquet(
            f"s3://aria-mapping/outputs/semantic_taxonomy/assignments/{label}.parquet"
        )
        cooccur_df.to_parquet(
            f"s3://aria-mapping/outputs/semantic_taxonomy/cooccurrences/{label}.parquet"
        )
