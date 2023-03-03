import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import argparse, boto3, pickle, random
from typing import Dict, Tuple, Any, Sequence
import numpy as np
import pandas as pd
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
)
from dap_aria_mapping.getters.taxonomies import get_entity_embeddings

METHODS = {
    "KMeans": KMeans,
    "AgglomerativeClustering": AgglomerativeClustering,
    "DBSCAN": DBSCAN,
    "HDBSCAN": HDBSCAN,
}

OUTPUT_DIR = "outputs/simulations/noise"


def get_configuration(args):
    args.cluster_method = args.cluster_method.replace("_gmm", "")
    args.cluster_method = args.cluster_method.replace("_pca", "")

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

    return [[METHODS[method_taxonomy], config_taxonomy]]


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
        "--n_simulations",
        type=int,
        default=1,
        help="Number of simulations to run",
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
    embeddings, embeddings_discarded = (
        get_entity_embeddings(discarded=False),
        get_entity_embeddings(discarded=True),
    )

    embeddings_discarded = embeddings_discarded.loc[
        ~embeddings_discarded.index.isin(embeddings.index)
    ]

    if args.test:
        embeddings, embeddings_discarded = (
            embeddings.iloc[:1000, :1000],
            embeddings_discarded.iloc[:1000, :1000],
        )
        args.cluster_method = args.cluster_method + "_test"

    ratio = len(embeddings_discarded) / len(embeddings)

    logger.info("Running clustering on embeddings")
    for percentage in np.logspace(-1, 0, 10):
        ratio_str = str(round(percentage * ratio, 2))[-2:]
        logger.info(
            "Noise level: " + str(round(percentage, 2)) + " - ratio: " + ratio_str
        )

        for simul in range(args.n_simulations):
            embeddings_df = pd.concat(
                [
                    embeddings,
                    embeddings_discarded.sample(
                        frac=percentage, random_state=args.seed
                    ),
                ]
            )

            cluster_configs = get_configuration(args)

            cluster_outputs, plot_dicts = run_clustering_generators(
                cluster_configs,
                embeddings_df,
                imbalanced=True if "imbalanced" in args.cluster_method else False,
            )

            if all([args.production, args.test is False]):
                logger.info("Saving clustering results to S3")
                s3 = boto3.client("s3")
                s3.put_object(
                    Body=pickle.dumps(cluster_outputs),
                    Bucket=BUCKET_NAME,
                    Key=f"s3://aria-mapping/{OUTPUT_DIR}/raw_outputs/semantic_{args.cluster_method}_ratio_{ratio_str}_simul_{simul}.pkl",
                )

            if "centroids" in args.cluster_method:
                logger.info("Reversing order of centroid labels")
                for output_dict in cluster_outputs:
                    for k, v in output_dict["labels"].items():
                        output_dict["labels"][k] = v[::-1]

            logger.info("Creating dataframe of clustering results")
            dataframe = pipe(
                cluster_outputs[-1],
                partial(
                    make_dataframe, label=f"_{args.cluster_method}", cumulative=True
                ),
                lambda df: df.rename(
                    columns={
                        k: "Level_{}".format(int(v) + 1)
                        for v, k in enumerate(df.columns)
                    }
                ),
                lambda df: df.reset_index()
                .rename(columns={"index": "Entity"})
                .assign(
                    Entity=lambda df: df["Entity"].str.replace(r"_\d+", "", regex=True)
                )
                .set_index("Entity"),
            )

            if "centroids" in args.cluster_method:
                dataframe = normalise_centroids(dataframe)

            if args.production:
                logger.info("Saving dataframe of clustering results to S3")
                dataframe.to_parquet(
                    f"s3://aria-mapping/{OUTPUT_DIR}/formatted_outputs/semantic_{args.cluster_method}_clusters_ratio_{ratio_str}_simul_{simul}.parquet"
                )
