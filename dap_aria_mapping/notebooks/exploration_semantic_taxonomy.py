# %% [markdown]
# Summary of completed work:
# - [X] Use several levels of resolution with baseline models
# - [X] Use several levels of resolution with baseline models, clustering within levels (enforces hierarchy)
# - [X] Reconstruct endograms for Agglomerative Clustering (enforces hierarchy)
# - [X] Cluster on centroids as the resolution decreases
# - [X] Cluster NERs to extract "TERMS", then iteratively cluster from mean embeddings (jobs taxonomy approach)
# - [X] Cluster based on a co-occurrence of clusterings (AFS approach)
# - [X] Cluster based on a co-occurrence of clusterings, clustering within levels (AFS approach)
# - [X] Add silhouette scores for validation.
# - [X] Compute distribution of sizes of resulting clusters.
#
# Summary of pending tasks:
# - [ ] Refactor code.
# - [ ] Include docstrings.
# - [ ] Comment exploratory jupytext file.
# - [ ] SPECTER embeddings.
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap
import re, os, json
from sentence_transformers import SentenceTransformer
from typing import Union, Dict, Sequence, Callable, Tuple, Generator
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from hdbscan import HDBSCAN
from itertools import product
from tqdm import tqdm
from toolz import pipe
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import chain, count
from functools import partial
from dap_aria_mapping.getters.openalex import get_openalex_entities
from exploration_utils import (
    get_sample,
    filter_entities,
    embed,
    run_clustering_generators,
    make_subplot_embeddings,
    make_dendrogram,
)

np.random.seed(42)
# %%

openalex_entities = pipe(
    get_openalex_entities(),
    partial(get_sample, score_threshold=60, num_articles=500),
    partial(filter_entities, min_freq=60, max_freq=95),
)
# %%
# Create embeddings
embeddings = pipe(
    openalex_entities.values(),
    lambda oa: chain(*oa),
    set,
    list,
    partial(embed, model="paraphrase-MiniLM-L6-v2"),
)

# Define UMAP
params = [
    ["n_neighbors", [5]],
    ["min_dist", [0.05]],
    ["n_components", [2]],
]

keys, permuts = ([x[0] for x in params], list(product(*[x[1] for x in params])))
param_perms = [{k: v for k, v in zip(keys, perm)} for perm in permuts]

for perm in param_perms:
    embeddings_2d = umap.UMAP(**perm).fit_transform(embeddings)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=1)
    fig.suptitle(f"{perm}")
    plt.show()


# %% [markdown]
# ### Strictly hierarchical clustering
# %%
cluster_configs = [
    [
        KMeans,
        [
            {"n_clusters": 3, "n_init": 2},
            {"n_clusters": 2, "n_init": 2},
            {"n_clusters": 2, "n_init": 2},
        ],
    ],
    [
        AgglomerativeClustering,
        [{"n_clusters": 3}, {"n_clusters": 2}, {"n_clusters": 2}],
    ],
]

cluster_outputs_s, plot_dicts = run_clustering_generators(cluster_configs, embeddings)

fig, axis = plt.subplots(2, 3, figsize=(24, 16), dpi=200)
for idx, (cdict, cluster) in enumerate(plot_dicts):
    _, lvl = divmod(idx, 3)
    make_subplot_embeddings(
        embeddings=embeddings_2d,
        clabels=[int(e) for e in cdict.values()],
        axis=axis.flat[idx],
        label=f"{cluster.__module__} {str(lvl)}",
        s=4,
    )
for output in cluster_outputs_s:
    print(
        "Silhouette score - {} clusters - {}: {}".format(
            output["model"].__module__,
            output["model"].get_params()["n_clusters"],
            round(output["silhouette"], 3),
        )
    )
# %% [markdown]
# ### Fuzzy hierarchical clustering
# %%

cluster_configs = [
    [KMeans, [{"n_clusters": [5, 10, 25, 50], "n_init": 5}]],
    [AgglomerativeClustering, [{"n_clusters": [5, 10, 25, 50]}]],
    [DBSCAN, [{"eps": [0.05, 0.1], "min_samples": [4, 8]}]],
    [HDBSCAN, [{"min_cluster_size": [4, 8], "min_samples": [4, 8]}]],
]

cluster_outputs_f, plot_dicts = run_clustering_generators(cluster_configs, embeddings)

fig, axis = plt.subplots(4, 4, figsize=(40, 40), dpi=200)
for idx, (cdict, cluster) in enumerate(plot_dicts):
    make_subplot_embeddings(
        embeddings=embeddings_2d,
        clabels=[int(e) for e in cdict.values()],
        axis=axis.flat[idx],
        label=f"{cluster.__module__}",
        cmap="gist_ncar",
    )
for output in cluster_outputs_f:
    print(
        "Silhouette score - {}: {}".format(
            output["model"].__module__, round(output["silhouette"], 3)
        )
    )
# %% [markdown]
# ### Using dendrograms from Agglomerative Clustering (enforces hierarchy)
# %%
cluster_configs = [[AgglomerativeClustering, [{"n_clusters": 100}]]]

cluster_outputs_d, plot_dicts = run_clustering_generators(cluster_configs, embeddings)

dendrogram = make_dendrogram(
    cluster_dict=cluster_outputs_d["labels"], model=cluster_outputs_d["model"]
)

fig, axis = plt.subplots(2, 3, figsize=(24, 16), dpi=200)
for i, (split, ax) in enumerate(zip(dendrogram[-7:-1], axis.flat)):
    tags = [
        [tag for tag in cluster_tags if tag < embeddings.shape[0]]
        for cluster_tags in split.values()
    ]
    level_clust = list(
        chain(
            *[
                list(zip(tags[cluster], [cluster] * len(tags[cluster])))
                for cluster in range(len(tags))
            ]
        )
    )
    level_clust = sorted(level_clust, key=lambda x: x[0])
    label_clust = [x[1] for x in level_clust]
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=label_clust, s=4)
    ax.set_title("Level {}".format(len(axis.flat) - i))
    silhouette = silhouette_score(embeddings, label_clust)
    print(
        "Silhouette score - dendrogram level {}: {}".format(
            len(axis.flat) - i, round(silhouette, 3)
        )
    )
plt.show()

# %% [markdown]
# ### Using centroids of Kmeans clustering for further clustering (à la job skill taxonomy)
# %%
cluster_configs = [
    [
        KMeans,
        [
            {"n_clusters": 400, "n_init": 2, "centroids": False},
            {"n_clusters": 100, "n_init": 2, "centroids": True},
            {"n_clusters": 5, "n_init": 2, "centroids": True},
        ],
    ],
]

cluster_outputs_c, plot_dicts = run_clustering_generators(
    cluster_configs, embeddings, embeddings_2d=embeddings_2d
)

# %%
fig, axis = plt.subplots(1, 3, figsize=(24, 8), dpi=200)
for idx, cdict in enumerate(cluster_outputs_c):
    if cdict["centroid_params"] is None:
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
            c=cdict["model"].labels_,
            s=cdict["centroid_params"]["sizes"],
        )
    print(f"Silhouette score ({idx}): {round(cdict['silhouette'], 3)}")

# %% [markdown]
# ### Analysis
# %%
import altair as alt


def make_analysis_dicts(cluster_outputs, model_module_name):
    return [
        [
            {k: "_".join(map(str, v)) for k, v in dictionary["labels"].items()},
            dictionary["model"],
        ]
        for dictionary in cluster_outputs
        if dictionary["model"].__module__ == model_module_name
    ]


def make_analysis_dicts_alt(cluster_outputs, model_module_name):
    return [
        [{k: v[-1] for k, v in dictionary["labels"].items()}, dictionary["model"]]
        for dictionary in cluster_outputs
        if dictionary["model"].__module__ == model_module_name
    ]


def make_analysis_dataframe(analysis_dicts, coltype=""):
    return pd.concat(
        [
            pd.DataFrame.from_dict(
                dictionary[0],
                orient="index",
                columns=[
                    "{}_{}_{}".format(
                        dictionary[1].__module__.replace(".", ""), idx, coltype
                    )
                ],
            )
            for idx, dictionary in enumerate(analysis_dicts)
        ],
        axis=1,
    )


def make_plots(analysis_df):
    return alt.concat(
        *[
            (
                alt.Chart(analysis_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        f"{column}:O",
                        sort=alt.EncodingSortField(
                            field=f"{column}", op="count", order="descending"
                        ),
                    ),
                    y="count()",
                )
            )
            for column in analysis_df.columns
        ]
    )


# %% Strictly hierarchical clustering KMEANS
make_plots(
    make_analysis_dataframe(
        make_analysis_dicts(cluster_outputs_s, "sklearn.cluster._kmeans")
    )
)
# %% Strictly hierarchical clustering Agglomerative
make_plots(
    make_analysis_dataframe(
        make_analysis_dicts(cluster_outputs_s, "sklearn.cluster._agglomerative")
    )
)
# %% Fuzzi hierarchical clustering KMEANS
make_plots(
    make_analysis_dataframe(
        make_analysis_dicts(cluster_outputs_f, "sklearn.cluster._kmeans")
    )
)
# %% Dendrograms
ls = []
for i, tree_split in enumerate(dendrogram[-7:-1]):
    tags = [
        [tag for tag in cluster_tags if tag < embeddings.shape[0]]
        for cluster_tags in tree_split.values()
    ]
    level_clust = list(
        chain(
            *[
                list(zip(tags[cluster], [cluster] * len(tags[cluster])))
                for cluster in range(len(tags))
            ]
        )
    )
    level_clust = sorted(level_clust, key=lambda x: x[0])
    label_clust = [x[1] for x in level_clust]
    ls.append(
        pd.DataFrame({"tree_lvl_{}".format(6 - i): label_clust}, index=embeddings.index)
    )
analysis_df_d = pd.concat(ls, axis=1)

make_plots(analysis_df_d)

# %% Centroids
make_plots(
    make_analysis_dataframe(
        make_analysis_dicts_alt(cluster_outputs_c, "sklearn.cluster._kmeans")
    )
)
# %%
meta_cluster_df = [
    make_analysis_dataframe(  # strict kmeans
        make_analysis_dicts_alt(cluster_outputs_s, "sklearn.cluster._kmeans"), "s"
    ),
    make_analysis_dataframe(  # strict agglomerative
        make_analysis_dicts_alt(cluster_outputs_s, "sklearn.cluster._agglomerative"),
        "s",
    ),
    make_analysis_dataframe(  # f kmeans
        make_analysis_dicts(cluster_outputs_f, "sklearn.cluster._kmeans"), "f"
    ),
    make_analysis_dataframe(  # f agglomerative
        make_analysis_dicts(cluster_outputs_f, "sklearn.cluster._agglomerative"), "f"
    ),
    analysis_df_d,
    make_analysis_dataframe(
        make_analysis_dicts_alt(cluster_outputs_c, "sklearn.cluster._kmeans"), "c"
    ),
]
cooccur_dict = {
    idx: {id: 0 for id in meta_cluster_df[0].index} for idx in meta_cluster_df[0].index
}
for df in meta_cluster_df:
    df.index = df.index.set_names(["tag"])
    df.reset_index(inplace=True)
    for col in [x for x in df.columns if x != "tag"]:
        df_tmp = df[[col, "tag"]]
        df_tmp = df_tmp.merge(df_tmp, on=col).reset_index()
        di = df_tmp.groupby("tag_x")["tag_y"].apply(list).to_dict()
        for k, v in di.items():
            for m in v:
                cooccur_dict[k][m] += 1
# %%
cooccur_df = pd.DataFrame.from_dict(cooccur_dict)
cooccur_df.to_parquet("cooccur_df.parquet")
# %%
cooccur_df.head(10)
# %% [markdown]
