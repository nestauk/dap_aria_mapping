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
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap
import json
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from hdbscan import HDBSCAN
from itertools import product
from toolz import pipe
from collections import defaultdict
from itertools import chain
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
from nesta_ds_utils.loading_saving.S3 import (
    upload_obj
)
bucket_name = 'aria-mapping'
np.random.seed(42)
# %%

openalex_entities = pipe(
    get_openalex_entities(),
    partial(get_sample, score_threshold=80, num_articles=500),
    partial(filter_entities, min_freq=60, max_freq=95),
)
# %%
# Create embeddings
embeddings = pipe(
    openalex_entities.values(),
    lambda oa: chain(*oa),
    set,
    list,
    partial(embed, model="sentence-transformers/allenai-specter"),
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
            {"n_clusters": 10, "n_init": 2},
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
            output["silhouette"],
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
            output["model"].__module__, output["silhouette"]
        )
    )
# %% [markdown]
# ### Using dendrograms from Agglomerative Clustering (enforces hierarchy)
# %%
cluster_configs = [[AgglomerativeClustering, [{"n_clusters": 100}]]]

cluster_outputs_d, plot_dicts = run_clustering_generators(cluster_configs, embeddings, build_dendrogram=True)

fig, axis = plt.subplots(2, 3, figsize=(24, 16), dpi=200)
for i, ax in zip(range(6), axis.flat):
    make_subplot_embeddings(
        embeddings=embeddings_2d,
        clabels=[int(e[i]) for e in cluster_outputs_d["labels"].values()],
        axis=ax,
        label=f"Level {i}",
        s=4,
    )

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
    print(f"Silhouette score ({idx}): {cdict['silhouette']}")

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
    return (
        pd.concat(
            [
                pd.DataFrame.from_dict(
                    dictionary[0],
                    orient="index",
                    columns=[
                        "{}_{}{}".format(
                            dictionary[1], idx, coltype #.__module__.replace(".", "")
                        )
                    ],
                )
                for idx, dictionary in enumerate(analysis_dicts)
            ],
            axis=1,
        )
        # .reset_index()
        # .rename(columns={"index": "tag"})
    )


def make_plots(analysis_df):
    return alt.hconcat(
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
            if column != "tag"
        ]
    )


# %% Strictly hierarchical clustering KMEANS
make_plots(
    make_analysis_dataframe(
        make_analysis_dicts(cluster_outputs_s, "sklearn.cluster._kmeans"), "_s_km"
    )
)
# %% Strictly hierarchical clustering Agglomerative
make_plots(
    make_analysis_dataframe(
        make_analysis_dicts(cluster_outputs_s, "sklearn.cluster._agglomerative"), "_s_agg"
    )
)
# %% Fuzzy hierarchical clustering KMEANS
make_plots(
    make_analysis_dataframe(
        make_analysis_dicts(cluster_outputs_f, "sklearn.cluster._kmeans"), "_f"
    )
)
# %% Alternative, which works for dendrograms
def make_dataframe(cluster_outputs, label, cumulative=True):
    d = defaultdict(list)
    num_levels = len(cluster_outputs["silhouette"])
    if cumulative:
        for level in range(num_levels):
            for k, v in cluster_outputs["labels"].items():
                d[k].append("_".join(map(str, v[:(level+1)])))
    else:
        for level in range(num_levels):
            for k, v in cluster_outputs["labels"].items():
                d[k].append(v[level])
    return pd.DataFrame.from_dict(
        d, 
        orient="index", 
        columns=[
            "{}_{}{}".format(
                cluster_outputs["model"], idx, label
            )
            for idx in range(num_levels)
        ]
    )

analysis_df_s_kmeans = make_dataframe(cluster_outputs_s[2], "_s_kmeans")
make_plots(analysis_df_s_kmeans)
# %%
analysis_df_d = make_dataframe(cluster_outputs_d, "")
make_plots(analysis_df_d)
# %% Centroids
analysis_df_c = make_dataframe(cluster_outputs_c[-1], "_centroid", cumulative=False)
make_plots(analysis_df_c)
# %% 
make_plots(
    make_analysis_dataframe(
        make_analysis_dicts_alt(cluster_outputs_c, "sklearn.cluster._kmeans"), "_c"
    )
)
# %% [markdown]
#### Silhouette Scores
# %%
results = defaultdict(list)
for idx in range(2,6,3):
    results[
        "{}_strict".format(cluster_outputs_s[idx]["model"].__module__.replace(".", ""))
    ] = cluster_outputs_s[idx]["silhouette"]
for idx in range(0,4,1):
    results[
        "{}_fuzzy".format(cluster_outputs_f[idx]["model"].__module__.replace(".", ""), idx)
    ].append(cluster_outputs_f[idx]["silhouette"][0])
for idx in range(4,8,1):
    results[
        "{}_fuzzy".format(cluster_outputs_f[idx]["model"].__module__.replace(".", ""), idx)
    ].append(cluster_outputs_f[idx]["silhouette"][0])
results["agglomerative_dendrogram"] = cluster_outputs_d["silhouette"]
results["kmeans_centroid"] = cluster_outputs_c[-1]["silhouette"][::-1]
results = {"_".join([k,str(id)]): e for k,v in results.items() for id, e in enumerate(v)}

silhouette_df = pd.DataFrame(results, index=["silhouette"]).T.sort_values(
    "silhouette", ascending=False
)
display(silhouette_df)
# %% [markdown]
# ### Meta Clustering
# %%
meta_cluster_df = (
    pd.concat(
        [
            make_dataframe(cluster_outputs_s[2], "_strict"), # kmeans strict
            make_dataframe(cluster_outputs_s[5], "_strict"), # agglomerative strict
            make_analysis_dataframe(  # kmeans fuzzy
                make_analysis_dicts(cluster_outputs_f, "sklearn.cluster._kmeans"), "_f"
            ),
            make_analysis_dataframe(  # agglomerative fuzzy
                make_analysis_dicts(
                    cluster_outputs_f, "sklearn.cluster._agglomerative"
                ),
                "_f",
            ),
            make_dataframe(cluster_outputs_d, ""),  # dendrograms
            make_dataframe(cluster_outputs_c[-1], "_centroid", cumulative=False), # centroids
        ],
        axis=1,
    )
    .reset_index()
    .rename(columns={"index": "tag"})
)

# Create dictionary of co-occurrences
dfnp = meta_cluster_df[[x for x in meta_cluster_df.columns if x != "tag"]].to_numpy()
m1 = np.tile(dfnp.T, (1, dfnp.shape[0])).flatten()
m2 = np.repeat(dfnp.T, dfnp.shape[0])
mf = np.where(m1 == m2, 1, 0)
cooccur_dict = np.sum(mf.reshape((dfnp.shape[1], dfnp.shape[0], dfnp.shape[0])), axis=0)
cooccur_dict = {
    k: {e: int(v) for e, v in zip(meta_cluster_df["tag"], cooccur_dict[i])}
    for i, k in enumerate(meta_cluster_df["tag"])
}
# %% Exports
with open("tmp/cooccurrences.json", "w") as f:
    json.dump(cooccur_dict, f)

cooccur_df = pd.DataFrame.from_dict(cooccur_dict)
# # cooccur_df.to_parquet("s3://aria-mapping/outputs/semantic_prototypes/cooccurrences/cooccur.parquet.gzip", compression="gzip")
cooccur_df.head(10)
# %%
# for key, value in list(zip(["cooccur_dict.json", "cooccur_df.parquet"], [cooccur_dict, cooccur_df])):
#     path_name = f"outputs/semantic_prototypes/cooccurrences/{key}"
#     upload_obj(obj=value, bucket=bucket_name, path_to=path_name)
# # %% Export Lists of Lists #[HACK]
# def get_next_repeated_level(label_lists, level, parent_list):
#     return [
#         [
#             [i for i in value if i in label_lists[level][k]]
#             for k in set(label_lists[level].keys())
#         ]
#         for value in parent_list
#     ]


def get_next_unique_level(label_lists, level, parent_list):
    z = []
    for value in parent_list:
        zf = []
        for k in set(
            label_lists[level].keys()
        ):  # ie loops over 0,1,2.. in child groups
            li = []
            for i in value:
                if i in label_lists[level][k]:
                    li.append(i)
            if len(li) > 0:
                zf.append(li)
        if len(zf) > 0:
            z.append(zf)
    return z


list_labels = {}
# # Strict hierarchical
# for idx, label in zip([2, -1], ["strict_kmeans", "strict_agglomerative"]):
#     labels = {
#         i: {k: v[i] for k, v in cluster_outputs_s[idx]["labels"].items()}
#         for i in range(3)
#     }
#     label_lists = {
#         i: {
#             group: [i for i, j in labels[i].items() if j == group]
#             for group in set(labels[i].values())
#         }
#         for i in range(3)
#     }
#     list_labels[label] = [
#         get_next_repeated_level(label_lists, 2, valuelist)
#         for valuelist in get_next_repeated_level(
#             label_lists, 1, label_lists[0].values()
#         )
#     ]

# # Dendogram
# cluster_outputs_dendrogram = dict(
#     zip(analysis_df_d.index, analysis_df_d.values.tolist())
# )
# labels = {i: {k: v[i] for k, v in cluster_outputs_dendrogram.items()} for i in range(6)}
# label_lists = {
#     i: {
#         group: [i for i, j in labels[i].items() if j == group]
#         for group in set(labels[i].values())
#     }
#     for i in range(6)
# }
# m0 = get_next_unique_level(label_lists, 4, label_lists[5].values())
# m1 = [get_next_unique_level(label_lists, 3, g) for g in m0]
# m2 = [[get_next_unique_level(label_lists, 2, i) for i in g] for g in m1]
# m3 = [[[get_next_unique_level(label_lists, 1, j) for j in i] for i in g] for g in m2]
# list_labels["dendrogram"] = [
#     [[[get_next_unique_level(label_lists, 0, e) for e in j] for j in i] for i in g]
#     for g in m3
# ]

# # Centroid clustering
# labels = {
#     i: {k: v[i] for k, v in cluster_outputs_c[-1]["labels"].items()} for i in range(3)
# }
# label_lists = {
#     i: {
#         group: [i for i, j in labels[i].items() if j == group]
#         for group in set(labels[i].values())
#     }
#     for i in range(3)
# }
# list_labels["centroid_kmeans"] = [
#     get_next_unique_level(label_lists, 0, valuelist)
#     for valuelist in get_next_unique_level(label_lists, 1, label_lists[2].values())
# ]

# # Export
# for key, value in list_labels.items():
#     path_name = f"outputs/semantic_prototypes/lists/{key}.json"
#     upload_obj(obj=value, bucket=bucket_name, path_to=path_name)
# %%
