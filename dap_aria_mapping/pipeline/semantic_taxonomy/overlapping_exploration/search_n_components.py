from dap_aria_mapping import PROJECT_DIR, BUCKET_NAME, logger
from typing import Sequence
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from dap_aria_mapping.getters.taxonomies import get_entity_embeddings
import matplotlib.pyplot as plt
import pickle, argparse, boto3
from tqdm import tqdm

FOLDER = PROJECT_DIR / "outputs" / "exploration" / "gmm"
FOLDER.mkdir(parents=True, exist_ok=True)


def plot_scores(
    n_components: Sequence[int],
    aic_scores: Sequence[float],
    bic_scores: Sequence[float],
    savename: str,
    args: argparse.Namespace,
) -> None:
    """Plot AIC and BIC scores for a range of GMM models.

    Args:
        n_components (int): The number of components used in each GMM.
        aic_scores (Sequence[float]): The AIC scores for each model.
        bic_scores (Sequence[float]): The BIC scores for each model.
        savename (str): The name to use when saving the plot.
    """
    min_bic, min_aic = (
        min(bic_scores, key=bic_scores.get),
        min(aic_scores, key=aic_scores.get),
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(n_components, aic_scores.values(), label="AIC")
    ax.plot(n_components, bic_scores.values(), label="BIC")
    ax.axvline(min_aic, linestyle="--")
    ax.axvline(min_bic, linestyle="--")
    ax.text(min_aic, 0, f"AIC = {min_aic}", fontsize=10, rotation=90)
    ax.text(min_bic, 0, f"BIC = {min_bic}", fontsize=10, rotation=90)
    ax.set(xlabel="Number of Topics", ylabel="Score")
    ax.legend(loc="best")
    if args.pca:
        fig.suptitle(f"PCA with {args.pca_max_components} components")
    if args.local:
        fig.savefig(FOLDER / f"{savename}.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore GMM models for entity embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--max_components",
        type=int,
        default=10_000,
        help="Maximum number of components to use in the GMM.",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=500,
        help="Step size for the number of components.",
    )

    parser.add_argument(
        "--pca",
        action="store_true",
        help="Whether to use PCA to reduce the dimensionality of the embeddings.",
    )

    parser.add_argument(
        "--pca_max_components",
        type=int,
        default=50,
        help="Maximum number of components to use in the PCA.",
    )

    parser.add_argument(
        "--pca_step",
        type=int,
        default=5,
        help="Step size for the number of components in the PCA.",
    )

    parser.add_argument(
        "--production",
        action="store_true",
        help="Whether to overwrite the outputs in the S3 bucket",
    )

    parser.add_argument(
        "--local",
        action="store_true",
        help="Whether to overwrite the outputs in the local folder",
    )

    args = parser.parse_args()

    logger.info("Loading entity embeddings.")
    embeddings = get_entity_embeddings()

    if not args.pca:
        n_components = np.arange(500, args.max_components, args.step)

        logger.info("Fitting GMM models.")
        models = {
            n: GaussianMixture(
                n_components=n,
                covariance_type="diag",
                random_state=42,
                verbose=1,
                reg_covar=1e-5,
            ).fit(embeddings)
            for n in tqdm(n_components)
        }

        logger.info("Saving GMM models.")
        if args.production:
            if args.local:
                with open(FOLDER / "gmm.pkl", "wb") as f:
                    pickle.dump(models, f)
            else:
                s3 = boto3.client("s3")
                s3.put_object(
                    Body=pickle.dumps(embeddings),
                    Bucket=BUCKET_NAME,
                    Key=f"outputs/semantic_taxonomy/gmm_outputs/gmm.pkl",
                )
        with open(FOLDER / "gmm.pkl", "wb") as f:
            pickle.dump(models, f)

        logger.info("Calculating BIC and AIC scores.")
        bic_scores, aic_scores = (
            {k: m.bic(embeddings) for k, m in models.items()},
            {k: m.aic(embeddings) for k, m in models.items()},
        )

        logger.info("Plotting BIC and AIC scores.")
        plot_scores(n_components, aic_scores, bic_scores, "gmm_scores")

    else:
        logger.info("Fitting PCA models.")
        for ncomp in range(5, args.pca_max_components, args.pca_step):
            pca = PCA(n_components=ncomp)
            embeddings_pca = pca.fit_transform(embeddings)

            logger.info(f"Fitting GMM models for PCA with {ncomp} components.")
            n_components = np.arange(500, args.max_components, args.step)
            models = {
                n: GaussianMixture(
                    n_components=n,
                    covariance_type="diag",
                    random_state=42,
                    verbose=1,
                    reg_covar=1e-5,
                ).fit(embeddings_pca)
                for n in tqdm(n_components)
            }

            logger.info(f"Saving GMM models for PCA with {ncomp} components.")
            if args.production:
                if args.local:
                    with open(FOLDER / f"gmm_pca_{str(ncomp)}.pkl", "wb") as f:
                        pickle.dump(models, f)
                else:
                    s3 = boto3.client("s3")
                    s3.put_object(
                        Body=pickle.dumps(embeddings),
                        Bucket=BUCKET_NAME,
                        Key=f"outputs/semantic_taxonomy/gmm_outputs/gmm_pca_{str(ncomp)}.pkl",
                    )

            logger.info(
                f"Calculating BIC and AIC scores for PCA with {ncomp} components."
            )
            bic_scores, aic_scores = (
                {k: m.bic(embeddings_pca) for k, m in models.items()},
                {k: m.aic(embeddings_pca) for k, m in models.items()},
            )

            logger.info(f"Plotting BIC and AIC scores for PCA with {ncomp} components.")
            plot_scores(
                n_components, aic_scores, bic_scores, f"gmm_pca_scores_{str(ncomp)}"
            )
