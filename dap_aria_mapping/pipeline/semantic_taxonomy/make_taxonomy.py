import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import argparse, boto3, pickle, random
from typing import Dict, Tuple, Any, Sequence
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
from toolz import pipe
from functools import partial
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from hdbscan import HDBSCAN
from dap_aria_mapping import PROJECT_DIR, BUCKET_NAME, logger, taxonomy
from dap_aria_mapping.utils.semantics import (
    make_dataframe,
    run_clustering_generators,
    normalise_centroids,
    make_subplot_embeddings,
)
from dap_aria_mapping.getters.taxonomies import get_entity_embeddings
from random import sample

METHODS = {
    "KMeans": KMeans,
    "AgglomerativeClustering": AgglomerativeClustering,
    "DBSCAN": DBSCAN,
    "HDBSCAN": HDBSCAN,
}

OUTPUT_DIR = "outputs/semantic_taxonomy"


def plot_centroids(cluster_outputs: Dict[str, Any]) -> plt.Figure:
    """Plot centroids for each level of clustering.

    Args:
        cluster_outputs (Tuple[Dict[str, Any], Sequence[Any]]): The output of the clustering
            function.
        embeddings_2d (np.ndarray): A two-dimensional embedding of the embeddings.

    Returns:
        plt.Figure: A figure with a subplot for each level of clustering.
    """
    num_levels = len(cluster_outputs[-1]["silhouette"])
    fig, axis = plt.subplots(1, num_levels, figsize=(int(num_levels * 8), 8), dpi=300)
    for idx, cdict in enumerate(cluster_outputs):
        if idx == 0:
            axis[idx].scatter(
                cdict["centroid_params"]["out_embeddings_2d"][:, 0],
                cdict["centroid_params"]["out_embeddings_2d"][:, 1],
                c=[e[0] for e in cdict["labels"].values()],
                s=1,
            )
        else:
            axis[idx].scatter(
                cdict["centroid_params"]["out_embeddings_2d"][:, 0],
                cdict["centroid_params"]["out_embeddings_2d"][:, 1],
                c=cdict["model"][idx].labels_,
                s=cdict["centroid_params"]["sizes"],
            )
        axis[idx].set_title(f"Centroids - Level {idx}")
    return fig


def plot_imbalanced(
    cluster_outputs: Dict[str, Any],
    plot_dicts: Sequence[Any],
    embeddings_2d: np.ndarray,
) -> plt.Figure:
    """Plot centroids for each level of clustering.

    Args:
        cluster_outputs (Tuple[Dict[str, Any], Sequence[Any]]): The output of the clustering
            function.
        embeddings_2d (np.ndarray): A two-dimensional embedding of the embeddings.

    Returns:
        plt.Figure: A figure with a subplot for each level of clustering.
    """
    num_levels = len(cluster_outputs[-1]["silhouette"])
    fig, axis = plt.subplots(1, num_levels, figsize=(int(num_levels * 8), 8), dpi=300)

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
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster_method",
        type=str,
        default="centroids",
        help="Which cluster config to use",
    )
    parser.add_argument(
        "--gmm",
        action="store_true",
        help="Whether to use GMM for centroids config",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Whether to use PCA ahead of GMM for centroids config",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the results (only centroids config)",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Whether to overwrite the taxonomy in the S3 bucket",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to run in test mode (only 1000 embeddings)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--sample_frac",
        type = float,
        action="store",
        help="percent of total entities to sample for sensitivity analysis"
    )

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.production and not any(
        ["centroids" in args.cluster_method, "imbalanced" in args.cluster_method]
    ):
        raise ValueError("Production mode only supports centroids config")

    logger.info("Downloading embeddings from S3")
    embeddings = get_entity_embeddings()
    if args.test:
        embeddings = embeddings.iloc[:1000, :1000]
        args.cluster_method = args.cluster_method + "_test"
    
    if args.sample_frac is not None:
        sample_size = int(args.sample_frac * len(embeddings))
        embeddings = sample(embeddings, k=sample_size)

    logger.info("Running UMAP on embeddings")
    embeddings_2d = umap.UMAP(
        n_neighbors=5, min_dist=0.05, n_components=2, random_state=args.seed
    ).fit_transform(embeddings)

    logger.info("Running clustering on embeddings")
    method_taxonomy, config_taxonomy = [
        taxonomy["SEMANTIC_TAXONOMY"][key][args.cluster_method]
        for key in ["CLUSTER_METHODS", "SINGLE_CLUSTER_CONFIGS"]
    ]

    if all([args.gmm, "centroids" in args.cluster_method]):
        args.cluster_method = args.cluster_method + "_gmm"
        if args.test:
            config_taxonomy[0].update({"gmm": True, "n_components": 100})
        else:
            config_taxonomy[0].update({"gmm": True, "n_components": 10_000})
        if args.pca:
            args.cluster_method = args.cluster_method + "_pca"
            config_taxonomy[0].update({"pca": True})

    cluster_configs = [[METHODS[method_taxonomy], config_taxonomy]]
    imbalanced = True if "imbalanced" in args.cluster_method else False

    cluster_outputs, plot_dicts = run_clustering_generators(
        cluster_configs,
        embeddings,
        imbalanced=imbalanced,
    )

    if all([args.production, args.test is False, args.sample_frac is None]):
        logger.info("Saving clustering results to S3")
        s3 = boto3.client("s3")
        s3.put_object(
            Body=pickle.dumps(cluster_outputs),
            Bucket=BUCKET_NAME,
            Key=f"{OUTPUT_DIR}/raw_outputs/semantic_{args.cluster_method}.pkl",
        )

    if args.plot:
        plot_folder = PROJECT_DIR / "outputs" / "figures" / "semantic_taxonomy"
        plot_folder.mkdir(parents=True, exist_ok=True)

        if "centroids" in args.cluster_method:
            logger.info("Creating plot of centroids clustering results")
            fig = plot_centroids(cluster_outputs)
        elif "imbalanced" in args.cluster_method:
            logger.info("Creating plot of imbalanced clustering results")
            fig = plot_imbalanced(cluster_outputs, plot_dicts, embeddings_2d)

        fig.savefig(plot_folder / f"semantic_{args.cluster_method}.png")

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

    if all([args.production, args.sample_frac is None]):
        logger.info("Saving dataframe of clustering results to S3")
        dataframe.to_parquet(
            f"s3://aria-mapping/{OUTPUT_DIR}/assignments/semantic_{args.cluster_method}_clusters.parquet"
        )
    
    if all([args.production, args.sample_frac is not None]):
        logger.info("Saving dataframe of clustering results to S3")
        dataframe.to_parquet(
            f"s3://aria-mapping/{OUTPUT_DIR}/assignments/semantic_{args.cluster_method}_{str(int(args.sample_frac*100))}_clusters.parquet"
        )
