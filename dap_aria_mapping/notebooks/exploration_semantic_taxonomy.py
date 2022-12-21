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
# The utils for this notebook include all necessary functions to create the taxonomy, including class ClusteringRoutine that performs any of the clustering methods described above. The function run_clustering_generators is used to run all clustering methods and return the results in a dictionary. The function make_dataframe is used to create a dataframe with the results of the clustering methods. The function make_plots is used to create a series of plots to visualize the results of the clustering methods. The function make_cooccurrences is used to create a co-occurrence matrix of the clustering results. The function make_subplot_embeddings is used to create a series of subplots with the embeddings of the entities in the taxonomy.

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from hdbscan import HDBSCAN
from itertools import product
from toolz import pipe
from collections import defaultdict
from itertools import chain
from functools import partial
from dap_aria_mapping import PROJECT_DIR
from dap_aria_mapping.getters.openalex import get_openalex_entities
from dap_aria_mapping.notebooks.exploration_utils import (
    get_sample,
    filter_entities,
    embed,
    make_subplot_embeddings,
    make_dataframe,
    make_plots,
    make_cooccurrences,
    run_clustering_generators
)

bucket_name = 'aria-mapping'
np.random.seed(42)
# %% [markdown]
# ### Obtain entities from OpenAlex
# The entity tags are obtained from OpenAlex. Filtering is applied to remove entities that are too frequent or too infrequent. The entities are then embedded using the SPECTER model. In addition, two-dimensional representations of the embeddings are obtained using UMAP, for plotting purposes.

openalex_entities = pipe(
    get_openalex_entities(),
    partial(get_sample, score_threshold=80, num_articles=3_000),
    partial(filter_entities, min_freq=60, max_freq=95),
)

# embeddings
embeddings = pipe(
    openalex_entities.values(),
    lambda oa: chain(*oa),
    set,
    list,
    partial(embed, model="sentence-transformers/allenai-specter"),
)
print(embeddings.shape)

# UMAP
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
# The following clustering routine iteratively clusters the entity embeddings using Agglomerative Clustering and KMeans. At each level, clustering is performed on the subsets that were created at the previous level.

cluster_configs = [
    [
        KMeans,
        [
            {"n_clusters": 3, "n_init": 5}, # parent level
            {"n_clusters": 2, "n_init": 5}, # nested level 1
            {"n_clusters": 10, "n_init": 5},# nested level 2
        ],
    ],
    [
        AgglomerativeClustering,
        [
            {"n_clusters": 3}, # parent level
            {"n_clusters": 2}, # nested level 1
            {"n_clusters": 2}  # nested level 2
        ],
    ],
]
# run clustering generators
cluster_outputs_s, plot_dicts = run_clustering_generators(cluster_configs, embeddings)
# %%
# plot results
fig, axis = plt.subplots(2, 3, figsize=(24, 16), dpi=200)
for idx, (cdict, cluster) in enumerate(plot_dicts):
    _, lvl = divmod(idx, 3)
    make_subplot_embeddings(
        embeddings=embeddings_2d,
        clabels=[int(e) for e in cdict.values()],
        axis=axis.flat[idx],
        label=f"{cluster[-1]} {str(lvl)}",
        s=4,
    )
fig.savefig(PROJECT_DIR / "dap_aria_mapping" / "notebooks" / "tmp" / "clustering_strict.png")

# print silhouettes
for output in cluster_outputs_s:
    print(
        "Silhouette score - {} clusters - {}: {}".format(
            output["model"][-1].__module__,
            output["model"][-1].get_params()["n_clusters"],
            output["silhouette"],
        )
    )
# %% [markdown]
# ### Strict hierarchical clustering with imbalanced nested clusters
# The following clustering routine iteratively clusters the entity embeddings using KMeans. At each level, clustering is performed on the subsets that were created at the previous level. The number of clusters at each level is allowed to vary, being determined by the size of the parent cluster.

cluster_configs = [
    [
        KMeans,
        [
            {"n_clusters": 3, "n_init": 5}, # parent level
            {"n_clusters": 10, "n_init": 5},# nested level 1, total n_clusters is 10+
            {"n_clusters": 10, "n_init": 5},# nested level 2, total n_clusters is 10+
        ],
    ],
]
# run clustering generators with imbalanced nested clusters
cluster_outputs_simb, plot_dicts = run_clustering_generators(cluster_configs, embeddings, imbalanced=True)

# %%
# plot results
fig, axis = plt.subplots(1, 3, figsize=(24, 8), dpi=200)
for idx, (cdict, cluster) in enumerate(plot_dicts):
    labels = [int(e) for e in cdict.values()]
    di = dict(zip(sorted(set(labels)), range(len(set(labels)))))
    labels = [di[label] for label in labels]
    _, lvl = divmod(idx, 3)
    make_subplot_embeddings(
        embeddings=embeddings_2d,
        clabels=labels,
        axis=axis.flat[idx],
        label=f"{cluster[-1]} {str(lvl)}",
        s=4,
    )
fig.savefig(PROJECT_DIR / "dap_aria_mapping" / "notebooks" / "tmp" / "clustering_strict_imb.png")

# print silhouettes
for output in cluster_outputs_simb:
    print(
        "Silhouette score - {} clusters - {}: {}".format(
            output["model"][-1].__module__,
            output["model"][-1].get_params()["n_clusters"],
            output["silhouette"],
        )
    )

# %% [markdown]
# ### Fuzzy hierarchical clustering
# The following approach iteratively clusters the entity embeddings using any sklearn method that supports the `predict_proba` method. No notion of level exists through this approach: more fine-grained clusterings are agnostic about the parent cluster output. Including several lists of parameter values will produce outputs for the Cartesian product of all parameter values within a clustering method.
# %%

cluster_configs = [
    [KMeans, [{"n_clusters": [5, 10, 25, 50], "n_init": 5}]], # level 1, level 2, level 3, level 4
    [AgglomerativeClustering, [{"n_clusters": [5, 10, 25, 50]}]], # level 1, level 2, level 3, level 4
    [DBSCAN, [{"eps": [0.05, 0.1], "min_samples": [4, 8]}]], # level 1, level 2, level 3, level 4
    [HDBSCAN, [{"min_cluster_size": [4, 8], "min_samples": [4, 8]}]], # level 1, level 2, level 3, level 4
]

# run clustering generators with fuzzy clusters
cluster_outputs_f_, plot_dicts = run_clustering_generators(cluster_configs, embeddings)

# %%
# plot results
fig, axis = plt.subplots(4, 4, figsize=(40, 40), dpi=200)
for idx, (cdict, cluster) in enumerate(plot_dicts):
    make_subplot_embeddings(
        embeddings=embeddings_2d,
        clabels=[int(e) for e in cdict.values()],
        axis=axis.flat[idx],
        label=f"{cluster[-1].__module__}",
        cmap="gist_ncar",
    )
fig.savefig(PROJECT_DIR / "dap_aria_mapping" / "notebooks" / "tmp" / "clustering_fuzzy.png")

# print silhouettes
for cluster in cluster_outputs_f_:
    print(
        "Silhouette score - {}: {}".format(
            cluster["model"][-1], cluster["silhouette"]
        )
    )

# %% [markdown]
# ### Using dendrograms from Agglomerative Clustering (enforces hierarchy)
# This approach uses a single run of any sklearn clustering method that supports a children_ attribute. The children_ attribute is used to recreate th dendrogram that produced the clustering, which is then used to create the taxonomy. The climbing algorithm advances one union of subtrees at a time. The number of levels is determined by the `dendrogram_levels` parameter.

cluster_configs = [[AgglomerativeClustering, [{"n_clusters": 100}]]]

# run clustering generators with dendrograms
cluster_outputs_d, plot_dicts = run_clustering_generators(cluster_configs, embeddings, dendrogram_levels=6)

# %%
# plot results
fig, axis = plt.subplots(2, 3, figsize=(24, 16), dpi=200)
for i, ax in zip(range(6), axis.flat):
    make_subplot_embeddings(
        embeddings=embeddings_2d,
        clabels=[int(e[i]) for e in cluster_outputs_d["labels"].values()],
        axis=ax,
        label=f"denrogram - level {i}",
        s=4,
    )
fig.savefig(PROJECT_DIR / "dap_aria_mapping" / "notebooks" / "tmp" / "clustering_dendrogram.png")
# %% [markdown]
# ### Centroids of Kmeans clustering as children nodes for further clustering (à la [job skill taxonomy](https://github.com/nestauk/skills-taxonomy-v2/tree/dev/skills_taxonomy_v2/pipeline/skills_taxonomy))
# This approach uses any number of nested KMeans clustering runs. After a given level, the centroids of the previous level are used as the new data points for the next level. 
# %%
cluster_configs = [
    [
        KMeans,
        [
            {"n_clusters": 400, "n_init": 5, "centroids": False},
            {"n_clusters": 200, "n_init": 5, "centroids": True},
            {"n_clusters": 20, "n_init": 5, "centroids": True},
            {"n_clusters": 5, "n_init": 5, "centroids": True},
        ],
    ],
]

# run clustering generators with centroids
cluster_outputs_c, plot_dicts = run_clustering_generators(
    cluster_configs, embeddings, embeddings_2d=embeddings_2d
)
# [HACK] flip order, should be fixed in run_clustering_generators (should run highest level → lowest level)
for output_dict in cluster_outputs_c:
    for k,v in output_dict["labels"].items():
        output_dict["labels"][k] = v[::-1]
    output_dict["silhouette"] = output_dict["silhouette"][::-1]

# %%
# plot results
fig, axis = plt.subplots(1, 4, figsize=(32, 8), dpi=200)
for idx, cdict in enumerate(cluster_outputs_c):
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
    print(f"Silhouette score ({idx}): {cdict['silhouette']}")
fig.savefig(PROJECT_DIR / "dap_aria_mapping" / "notebooks" / "tmp" / "clustering_centroids.png")
# %% [markdown]
# ### Analysis
# This section outputs silhouette scores for all relevant outputs above. It also constructs barplots of the cluster sizes for each level of the taxonomy across approaches. 
# %%
# Harmonize cluster outputs for analysis
# [HACK] - fix this. For exports, I create a single dictionary for the fuzzy clusters
cluster_outputs_f = []
for group in ["sklearn.cluster._kmeans", "sklearn.cluster._agglomerative"]:
    dict_group = {
        "labels": defaultdict(list),
        "model": [],
        "silhouette": [],
        "centroid_params": None
    }

    cluster_outpu = [x for x in cluster_outputs_f_ if x["model"][0].__module__ == group]
    for clust in cluster_outpu:
        for k, v in clust["labels"].items():
            dict_group["labels"][k].append(v[0])
        dict_group["model"].append("_".join([clust["model"][0].__module__.replace(".", ""), str(clust["model"][0].get_params()["n_clusters"])]))
        dict_group["silhouette"].append(clust["silhouette"][0])
    cluster_outputs_f.append(dict_group)

# %% Create dataframes
strict_kmeans_df = make_dataframe(cluster_outputs_s[2], "_strict")
strict_agglom_df = make_dataframe(cluster_outputs_s[5], "_strict")
strict_kmeans_imb_df = make_dataframe(cluster_outputs_simb[-1], "_strict_imbalanced")
fuzzy_kmeans_df = make_dataframe(cluster_outputs_f[0], "_fuzzy")
fuzzy_agglom_df = make_dataframe(cluster_outputs_f[1], "_fuzzy")
dendrogram_df = make_dataframe(cluster_outputs_d, "")
centroid_kmeans_df = make_dataframe(cluster_outputs_c[-1], "_centroids", cumulative=False)
# %% Strictly hierarchical clustering KMEANS
make_plots(strict_kmeans_df)
# %% Strictly hierarchical clustering Agglomerative
make_plots(strict_agglom_df)
# %% Strictly hierarchical clustering KMEANS with imbalanced cluster numbers
make_plots(strict_kmeans_imb_df)
# %% Fuzzy hierarchical clustering KMEANS
make_plots(fuzzy_kmeans_df)
# %% Dendrogram
make_plots(dendrogram_df)
# %% Centroids using Kmeans
make_plots(centroid_kmeans_df)
# %% [markdown]
# #### Silhouette Scores
# %% 
results = {
    "kmeans_strict": cluster_outputs_s[2]["silhouette"],
    "agglom_strict": cluster_outputs_s[5]["silhouette"],
    "kmeans_strict_imb": cluster_outputs_simb[-1]["silhouette"],
    "kmeans_fuzzy": cluster_outputs_f[0]["silhouette"],
    "agglom_fuzzy": cluster_outputs_f[1]["silhouette"],
    "agglomerative_dendrogram": cluster_outputs_d["silhouette"],
    "kmeans_centroid": cluster_outputs_c[-1]["silhouette"],
}

results = {"_".join([k,str(id)]): e for k,v in results.items() for id, e in enumerate(v)}

silhouette_df = pd.DataFrame(results, index=["silhouette"]).T.sort_values(
    "silhouette", ascending=False
)
silhouette_df.to_pickle(PROJECT_DIR / "dap_aria_mapping" / "notebooks" / "tmp" / "silhouette_df.pkl")
display(silhouette_df)
# %% [markdown]
# ### Meta Clustering
# Following the approach of Juan in the [AFS repository](https://github.com/nestauk/afs_neighbourhood_analysis/tree/1b1f1b1dabbebd07e5c85c72d7401107173bf863/afs_neighbourhood_analysis/analysis), we combine the clustering methods to produce a matrix of entity co-occurrences. The objective is to apply community detection algorithms on this.
# %%
list_dfs = [
    strict_kmeans_df, 
    strict_agglom_df,
    strict_kmeans_imb_df, 
    fuzzy_kmeans_df, 
    fuzzy_agglom_df,
    dendrogram_df, 
    centroid_kmeans_df
]

meta_cluster_df = (
    pd.concat(list_dfs, axis=1)
    .reset_index()
    .rename(columns={"index": "tag"})
)

# %% [markdown]
# ### Exports
# %%
for label, df in zip(
    [
        "strict_kmeans", 
        "strict_agglom",
        "strict_kmeans_imb", 
        "fuzzy_kmeans", 
        "fuzzy_agglom", 
        "dendrogram", 
        "centroid_kmeans", 
        "meta_cluster"
    ],
    list_dfs + [meta_cluster_df]
):
    cooccur_dict = make_cooccurrences(meta_cluster_df)
    cooccur_df = pd.DataFrame.from_dict(cooccur_dict)
    df.to_parquet(f"s3://aria-mapping/outputs/semantic_prototypes/assignments/{label}.parquet")
    cooccur_df.to_parquet(f"s3://aria-mapping/outputs/semantic_prototypes/cooccurrences/{label}.parquet")
cooccur_df.head(10)
# %%