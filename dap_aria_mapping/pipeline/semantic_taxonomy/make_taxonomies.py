"""
Creates taxonomies for all candidate semantic methods.
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import boto3, pickle
import numpy as np
import pandas as pd
import umap.umap_ as umap
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

    logger.info("Running clustering generators - Strict hierarchical clustering")
    cluster_outputs_s, _ = run_clustering_generators(
        CLUSTER_CONFIGS["strict"], embeddings
    )

    logger.info(
        "Running clustering generators - Strict hierarchical clustering with imbalanced clusters"
    )
    cluster_outputs_simb, _ = run_clustering_generators(
        CLUSTER_CONFIGS["strict_imb"], embeddings, imbalanced=True
    )

    logger.info("Running clustering generators - Fuzzy hierarchical clustering")
    cluster_outputs_f_, _ = run_clustering_generators(
        CLUSTER_CONFIGS["fuzzy"], embeddings
    )

    logger.info("Running clustering generators - Dendrogram")
    cluster_outputs_d, _ = run_clustering_generators(
        CLUSTER_CONFIGS["dendrogram"], embeddings, dendrogram_levels=200
    )

    logger.info(
        "Running clustering generators - Centroids of Kmeans clustering as children nodes for"
        " further clustering"
    )
    cluster_outputs_c, _ = run_clustering_generators(
        CLUSTER_CONFIGS["centroids"], embeddings, embeddings_2d=embeddings_2d
    )

    logger.info("Running clustering generators - Done")

    logger.info("Adjusting outputs from generator objects")
    for output_dict in cluster_outputs_c:
        for k, v in output_dict["labels"].items():
            output_dict["labels"][k] = v[::-1]
        output_dict["silhouette"] = output_dict["silhouette"][::-1]

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

    logger.info("Running dictionary & table exports")
    outputs_ls = [
        cluster_outputs_s[3],
        cluster_outputs_s[7],
        cluster_outputs_simb[-1],
        cluster_outputs_f[0],
        cluster_outputs_f[1],
        cluster_outputs_d,
        cluster_outputs_c[-1],
    ]

    outputs_df = []
    for dict_, df_config in zip(outputs_ls, df_configs):
        name_export = df_config["name"]
        s3.put_object(
            Body=pickle.dumps(dict_),
            Bucket=BUCKET_NAME,
            Key=f"outputs/semantic_taxonomy/raw_outputs/{name_export}.pkl",
        )

        df = make_dataframe(dict_, df_config["label"], df_config["cumulative"])
        if df_config["name"] == "agglom_dendrogram":
            depth_dendrogram = df.shape[1]
            levels = [int(depth_dendrogram * i) for i in [0.1, 0.3, 0.5, 0.7, 0.9]]
            df = df.iloc[:, levels]
        outputs_df.append(df)

    logger.info("Running silhouette score export")
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

    logger.info("Creating meta_cluster object")
    meta_cluster_df = pd.concat(outputs_df, axis=1)

    logger.info("Running cooccurrence matrix export")
    labels = [x["name"] for x in df_configs]
    for label, df in zip(labels + ["meta_cluster"], outputs_df + [meta_cluster_df]):
        logger.info(f"Running export routine for {label}")
        df = (
            df.rename(
                columns={k: "Level_{}".format(v + 1) for v, k in enumerate(df.columns)}
            )
            .reset_index()
            .rename(columns={"index": "Entity"})
            .set_index("Entity")
        )
        logger.info(f"Making co-occurrences for {label}")
        cooccur_dict = make_cooccurrences(df)
        logger.info(f"Transforming to co-occurrence dataframe ({label})")
        cooccur_df = pd.DataFrame(cooccur_dict, index=df.index, columns=df.index)

        logger.info(f"Saving parquets for {label}")
        df.to_parquet(
            f"s3://aria-mapping/outputs/semantic_taxonomy/assignments/{label}.parquet"
        )
        cooccur_df.to_parquet(
            f"s3://aria-mapping/outputs/semantic_taxonomy/cooccurrences/{label}.parquet"
        )
