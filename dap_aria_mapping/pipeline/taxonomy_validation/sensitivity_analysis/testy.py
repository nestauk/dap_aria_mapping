# %%
from dap_aria_mapping import PROJECT_DIR
from sklearn.mixture import GaussianMixture
import numpy as np
from dap_aria_mapping.getters.taxonomies import get_entity_embeddings
import matplotlib.pyplot as plt
import pickle

embeddings = get_entity_embeddings()
embeddings = embeddings[:100]
# n_components = np.arange(500, 10_000, 250)
n_components = np.arange(1, 5, 1)
models = {
    n: GaussianMixture(
        n_components=n, covariance_type="full", random_state=42, verbose=1
    ).fit(embeddings)
    for n in n_components
}

with open(PROJECT_DIR / "outputs" / "gmm.pkl", "wb") as f:
    pickle.dump(models, f)

bic_scores, aic_scores = (
    {k: m.bic(embeddings) for k, m in models.items()},
    {k: m.aic(embeddings) for k, m in models.items()},
)

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

fig.save(PROJECT_DIR / "outputs" / "gmm.png", dpi=300)
# %%
