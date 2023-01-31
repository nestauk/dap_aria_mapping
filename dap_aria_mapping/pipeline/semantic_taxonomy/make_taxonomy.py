import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import argparse, boto3, pickle, random
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
from toolz import pipe
from functools import partial
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from hdbscan import HDBSCAN
from dap_aria_mapping import PROJECT_DIR, BUCKET_NAME, logger, config
from dap_aria_mapping.utils.semantics import (
    make_dataframe,
    run_clustering_generators,
    normalise_centroids,
    make_subplot_embeddings,
)
from dap_aria_mapping.getters.taxonomies import get_embeddings

METHODS = {
    "KMeans": KMeans,
    "AgglomerativeClustering": AgglomerativeClustering,
    "DBSCAN": DBSCAN,
    "HDBSCAN": HDBSCAN,
}

OUTPUT_DIR = "outputs/semantic_taxonomy"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster_method",
        type=str,
        default="centroids",
        help="Which cluster config to use",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the results (only centroids config)",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Whether to run in production mode",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.production and not any(
        ["centroids" in args.cluster_method, "imbalanced" in args.cluster_method]
    ):
        raise ValueError("Production mode only supports centroids config")

    logger.info("Downloading embeddings from S3")
    embeddings = get_embeddings()
    if not args.production:
        embeddings = embeddings.iloc[:1000, :1000]

    logger.info("Running UMAP on embeddings")
    embeddings_2d = umap.UMAP(
        n_neighbors=5, min_dist=0.05, n_components=2
    ).fit_transform(embeddings)

    logger.info("Running clustering on embeddings")
    method_taxonomy, config_taxonomy = [
        config["SEMANTIC_TAXONOMY"][key][args.cluster_method]
        for key in ["CLUSTER_METHODS", "SINGLE_CLUSTER_CONFIGS"]
    ]
    cluster_configs = [[METHODS[method_taxonomy], config_taxonomy]]

    cluster_outputs, plot_dicts = run_clustering_generators(
        cluster_configs, embeddings, embeddings_2d=embeddings_2d
    )

    if args.production:
        logger.info("Saving clustering results to S3")
        s3 = boto3.client("s3")
        s3.put_object(
            Body=pickle.dumps(cluster_outputs),
            Bucket=BUCKET_NAME,
            Key=f"{OUTPUT_DIR}/raw_outputs/semantic_{args.cluster_method}.pkl",
        )

    if args.plot:
        if "centroids" in args.cluster_method:
            logger.info("Creating plot of centroids clustering results")
            num_levels = len(cluster_outputs[-1]["silhouette"])
            fig, axis = plt.subplots(
                1, num_levels, figsize=(int(num_levels * 8), 8), dpi=300
            )
            for idx, cdict in enumerate(cluster_outputs):
                if not cdict.get("centroid_params", False):
                    axis[idx].scatter(
                        embeddings_2d[:, 0],
                        embeddings_2d[:, 1],
                        c=[e for e in cdict["labels"].values()],
                        s=1,
                    )
                else:
                    axis[idx].scatter(
                        cdict["centroid_params"]["n_embeddings_2d"][:, 0],
                        cdict["centroid_params"]["n_embeddings_2d"][:, 1],
                        c=cdict["model"][idx].labels_,
                        s=cdict["centroid_params"]["sizes"],
                    )
        elif "imbalanced" in args.cluster_method:
            logger.info("Creating plot of imbalanced clustering results")
            num_levels = len(cluster_outputs[-1]["silhouette"])
            fig, axis = plt.subplots(
                1, num_levels, figsize=(int(num_levels * 8), 8), dpi=300
            )

            for idx, (cdict, cluster) in enumerate(plot_dicts):
                labels = [int(e) for e in cdict.values()]
                di = dict(zip(sorted(set(labels)), range(len(set(labels)))))
                labels = [di[label] for label in labels]
                _, lvl = divmod(idx, num_levels)
                make_subplot_embeddings(
                    embeddings=embeddings_2d,
                    clabels=labels,
                    axis=axis.flat[idx],
                    label=f"{cluster[-1]} {str(lvl)}",
                    s=4,
                )
        if args.production:
            fig.savefig(
                PROJECT_DIR
                / "outputs"
                / "figures"
                / "semantic_taxonomy"
                / f"semantic_{args.cluster_method}.png"
            )
        else:
            fig.savefig(
                PROJECT_DIR
                / "outputs"
                / "figures"
                / "semantic_taxonomy"
                / f"semantic_{args.cluster_method}_test.png"
            )

    if "centroids" in args.cluster_method:
        logger.info("Reversing order of centroid labels")
        for output_dict in cluster_outputs:
            for k, v in output_dict["labels"].items():
                output_dict["labels"][k] = v[::-1]

    logger.info("Creating dataframe of clustering results")
    dataframe = pipe(
        cluster_outputs[-1],
        partial(make_dataframe, label=f"_{args.cluster_method}", cumulative=True),
        lambda df: df.rename(
            columns={k: "Level_{}".format(int(v) + 1) for v, k in enumerate(df.columns)}
        ),
        lambda df: df.reset_index()
        .rename(columns={"index": "Entity"})
        .set_index("Entity"),
    )

    if "centroids" in args.cluster_method:
        dataframe = normalise_centroids(dataframe)

    if args.production:
        logger.info("Saving dataframe of clustering results to S3")
        dataframe.to_parquet(
            f"s3://aria-mapping/{OUTPUT_DIR}/assignments/semantic_{args.cluster_method}_clusters.parquet"
        )
