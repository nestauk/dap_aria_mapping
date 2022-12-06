# %% [markdown]
# Summary of pending tasks:
# - Generate a set of all DBpedia entities associated with the corpora (AI-genomics for now, files with db entities in a json)
# - Filter entities (e.g. those with very low frequency)
#
# - Embed each entity separately using a transformer (options could include sbert STS stransformers or SPECTER)
# - Use hierarchical or iterative clustering methods to generate clusters of entities at different levels of granularity.
# - This could use agglomerative clustering, HDBSCAN (with reassignment of outliers) or a method that requires a specified n_clusters like K means, GMM etc.
# - Probably worth using some clustering optimisation with e.g. silhouette score as a minimum.
#
# Rapid-fire ideas:
# - The clusters will be used to generate a semantic taxonomy.
# - The clusters could be visualised using UMAP or t-SNE.
# - The clusters could be used to generate a semantic recommender system.
# - The clusters could be used to generate a semantic knowledge graph.
#
# #### Discussion over pending tasks:
# ##### Clustering methods
# - Agglomerative clustering (sklearn)
# - HDBSCAN
#   - Reassignment of outliers
#   - Different cuts of the tree, as described in the [advanced methods section](https://hdbscan.readthedocs.io/en/latest/advanced_hdbscan.html)
# - K means (sklearn, or [FAISS](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization)?)
# - GMM (?) soft-clustering feels ill-adviced for the purposes of the work here.
#
# These methods can either be used as standalone, or combined in a network of cluster co-occurrences. Community detection algorithms (Louvain) can then be used to identify communities of these "joint" clusters. See Juan's [work](https://github.com/nestauk/afs_neighbourhood_analysis/blob/1b1f1b1dabbebd07e5c85c72d7401107173bf863/afs_neighbourhood_analysis/pipeline/lad_clustering/cluster_utils.py) on AFS.
#
# An alternative is to embedd on the average embeddings of the articles, efficiently storing a reverse mapping to the original embeddings. This arguably combines ideas of community & semantic neighborhoods.
# ##### Embedding methods
# - [SPECTER](https://github.com/nestauk/ai_genomics/tree/dev/ai_genomics/pipeline/description_embed)
# - [SBERT](https://www.sbert.net)
#
# These methods can be found [here](https://github.com/nestauk/ai_genomics/blob/dev/ai_genomics/pipeline/entity_cluster/create_entity_clusters.py), [here](https://github.com/nestauk/ai_genomics/blob/7d4e9301449dae5e0f2671d866ede445310a6b1b/ai_genomics/utils/entities.py#L28), and [here](https://github.com/nestauk/ai_genomics/blob/dev/ai_genomics/pipeline/entity_cluster/__init__.py).

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap
import re, os, json
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from hdbscan import HDBSCAN
from itertools import product
from tqdm import tqdm
from toolz import pipe
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import chain
from functools import partial
from dap_aria_mapping.getters.openalex import get_openalex_entities

np.random.seed(42)


def get_sample(entities, num_articles):
    draws = num_articles / len(entities.values()) <= np.random.uniform(
        0, 1, len(entities.values())
    )
    sample = {k: entities[k] for i, k in enumerate(entities.keys()) if draws[i]}
    return {k: [e[0] for e in v] for k, v in sample.items()}


def filter_entities(entities, min_freq=0, max_freq=100):
    assert min_freq < max_freq, "min_freq must be less than max_freq"
    frequencies = Counter(chain(*entities.values()))
    min_freq, max_freq = (
        np.percentile(list(frequencies.values()), min_freq),
        np.percentile(list(frequencies.values()), max_freq),
    )
    return {
        k: [e for e in v if min_freq <= frequencies[e] <= max_freq]
        for k, v in entities.items()
    }


# %%
openalex_entities = pipe(
    get_openalex_entities(),
    partial(get_sample, num_articles=10_000),
    partial(filter_entities, min_freq=90, max_freq=95),
)
# %% Embeddings
from sentence_transformers import SentenceTransformer


def embed(entities, model):
    model = SentenceTransformer(model)
    return pd.DataFrame(model.encode(entities), index=entities)


# Create embeddings
embeddings = pipe(
    openalex_entities.values(),
    lambda oa: chain(*oa),
    set,
    list,
    partial(embed, model="paraphrase-MiniLM-L6-v2"),
)

# %% Plot embedidngs

params = [
    ["n_neighbors", [2, 4, 6, 8, 10]],
    ["min_dist", [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]],
    ["n_components", [2, 3, 4, 5, 6, 7, 8, 9, 10]],
]

keys, permuts = ([x[0] for x in params], list(product(*[x[1] for x in params])))
param_perms = [{k: v for k, v in zip(keys, perm)} for perm in permuts]

for perm in param_perms:
    embeddings_2d = umap.UMAP(**perm).fit_transform(embeddings)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=1)
    fig.suptitle(f"{perm}")
    plt.show()

# %%
embeddings_2d = umap.UMAP().fit_transform(embeddings)
# %% Plot example clustering output


km = KMeans(n_clusters=100)
km.fit(embeddings)
clusters = dict(zip(embeddings.index, [int(l) for l in km.labels_]))

plt.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1], c=[int(l) for l in km.labels_], s=0.1
)
# %% [markdown]
# ### Exploration of baseline clustering methods
# %%


# %% [markdown]
# - [X] Use several levels of resolution with baseline models
# %%
def get_cluster_config(cmethod):
    _, cparams = cmethod
    keys, permuts = ([x[0] for x in cparams], list(product(*[x[1] for x in cparams])))
    for perm in permuts:
        yield {k: v for k, v in zip(keys, perm)}


def get_clusters(embeddings, cmethod):
    cclass, _ = cmethod
    cconfig = get_cluster_config(cmethod)
    for config in cconfig:
        cluster = cclass(**config)
        cluster.fit(embeddings)
        yield cluster, config


def get_param_initial(query, key):
    if m := re.match(query, key):
        return m.group(2)


def make_subplot_embeddings(embeddings, clabels, axis, label=None):
    axis.scatter(embeddings[:, 0], embeddings[:, 1], c=clabels, s=0.1)
    if label is not None:
        axis.set_title(label)


def make_dict_assignments(embeddings, model, config):
    dict_key = (
        get_param_initial(r"(.*\.)?(.*)", model.__module__)
        + "_"
        + "_".join(
            [get_param_initial(r"(.*_)?(.*)", i) + str(j) for i, j in config.items()]
        )
    )
    return {f"{dict_key}": dict(zip(embeddings.index, model.labels_))}


def get_cluster_outputs(cgens, cdict={}, figr=1, figc=1, figs=(10, 10)):
    (fig, axis), idx = plt.subplots(figr, figc, figsize=figs, dpi=200), 0
    for cgen in tqdm(cgens):
        for cluster, config in cgen:
            cdict.update(make_dict_assignments(embeddings, cluster, config))
            make_subplot_embeddings(
                embeddings=embeddings_2d,
                clabels=[int(e) for e in cluster.labels_],
                axis=axis.flat[idx],
                label=f"{cluster.__module__} {config}",
            )
            idx += 1  # [HACK] :(
    return cdict, fig


# Create cluster generator objects
# cluster_methods = [
#     [KMeans, [["n_clusters", [5, 10, 25, 50]], ["n_init", [5]]]],
#     [AgglomerativeClustering, [["n_clusters", [5, 10, 25, 50]]]],
#     [DBSCAN, [["eps", [0.05, 0.1]], ["min_samples", [4, 8]]]],
#     [HDBSCAN, [["min_cluster_size", [4, 8]], ["min_samples", [4, 8]]]]
# ]

# cluster_gens = [
#     get_clusters(embeddings, cmethod)
#     for cmethod in cmethods
# ]
# cluster_dict, cluster_plot = get_cluster_outputs(cluster_gens, figr=4, figc=4, figs=(40, 40))

# %% [markdown]
# - [X] Create a function that takes a clustering output and returns a list of the entities in each cluster.
# %%
def cluster_to_list(cluster, cdict):
    return [e for e in embeddings.index if cdict[e] == cluster]


# output_clusters = {
#     cluster: {
#         i: cluster_to_list(i, cdict[cluster])
#         for i in sorted(set(cdict[cluster].values()), reverse=False)
#     }
#     for cluster in cdict.keys()
# }
# %% [markdown]
# - [X] Use several levels of resolution with baseline models, clustering within levels (enforces hierarchy)
# %%


def get_cluster_config(params):
    keys, permuts = ([x[0] for x in params], list(product(*[x[1] for x in params])))
    for perm in permuts:
        yield {k: v for k, v in zip(keys, perm)}


def get_nested_cluster_config(nested_config):
    for child_params in nested_config:
        yield get_cluster_config(child_params)


def update_dictionary(cdict, embeddings, cluster):
    for key, date in zip(embeddings.index, cluster.labels_):
        cdict[key].append(date)


def clustering_routine(method_class, config, embeddings, nested=False):
    for child_config in config:
        for param_config in child_config:
            if nested is False:
                cluster = method_class(**param_config)
                cluster.fit(embeddings)
                cdict = defaultdict(list)
                update_dictionary(cdict, embeddings, cluster)
                yield deepcopy(cdict), cluster

            else:
                clusters = list(
                    list(e) for e in set([tuple(i) for i in cdict.values()])
                )
                for cluster_group in clusters:
                    nested_embeddings = embeddings.loc[
                        embeddings.index.isin(
                            [e for e in embeddings.index if cdict[e] == cluster_group]
                        )
                    ]

                    cluster = method_class(**param_config)
                    cluster.fit(nested_embeddings)
                    update_dictionary(cdict, nested_embeddings, cluster)
                yield deepcopy(cdict), cluster
        nested = True


def get_nested_clusters(cmethod, embeddings, nested=False):
    cclass, cconfig = cmethod
    nested_cconfig = get_nested_cluster_config(cconfig)
    return clustering_routine(cclass, nested_cconfig, embeddings, nested=nested)


def make_dict_key(model, config):
    return (
        get_param_initial(r"(.*\.)?(.*)", model.__module__)
        + "_"
        + "_".join(
            [get_param_initial(r"(.*_)?(.*)", i) + str(j) for i, j in config.items()]
        )
    )


cluster_configs = [
    [
        KMeans,
        [
            [["n_clusters", [10]], ["n_init", [5]]],
            [["n_clusters", [5]], ["n_init", [5]]],
        ],
    ],
    [AgglomerativeClustering, [[["n_clusters", [10]]], [["n_clusters", [5]]]]],
]

nested_cluster_gens = [
    get_nested_clusters(cluster_config, embeddings)
    for cluster_config in cluster_configs
]

cluster_dicts = [
    tuple([cdict, cluster]) for ncgen in nested_cluster_gens for cdict, cluster in ncgen
]

clusters_to_plot = [
    [{k: int("".join(map(str, v))) for k, v in clusters[0].items()}, clusters[1]]
    for clusters in cluster_dicts
]

fig, axis = plt.subplots(2, 2, figsize=(20, 20), dpi=200)


def make_subplot_embeddings(embeddings, clabels, axis, label=None, **kwargs):
    axis.scatter(embeddings[:, 0], embeddings[:, 1], c=clabels, s=1, **kwargs)
    if label is not None:
        axis.set_title(label)


for idx, (cdict, cluster) in enumerate(clusters_to_plot):
    make_subplot_embeddings(
        embeddings=embeddings_2d,
        clabels=[int(e) for e in cdict.values()],
        axis=axis.flat[idx],
        label=f"{cluster.__module__}",
        cmap="gist_ncar",
    )

# for cluster in clusters_to_plot:
#     vals = set(cluster[0].values())
#     d = dict(zip(sorted(vals), range(len(vals))))
#     cluster[0] = {k: d[v] for k, v in cluster[0].items()}

#
# %%
# %%
# %% [markdown]
# - [X] Use several levels of resolution with baseline models
# - [X] Use several levels of resolution with baseline models, clustering within levels (enforces hierarchy)
# - [ ] Cluster on centroids as the resolution decreases
# - [ ] Cluster NERs to extract "TERMS", then iteratively cluster from mean embeddings (jobs taxonomy approach)
# - [ ] Cluster based on a co-occurrence of clusterings (AFS approach)
# - [ ] Cluster based on a co-occurrence of clusterings, clustering within levels (AFS approach)
# %% [markdown]
# TODO:

# - [X] Create a generator that produces clusters at increasingly large
# - [X] Create dynamic plot generation.
# - [ ] Create co-occurrence clustering matrix.
# - [ ] Use matrix to greedily link terms.
# - [ ] Use matrix to build community detection (this is actually the same as greedy linking approaches, see Girvan-Newman).
# - [ ] Create embeddings from "document vectors" (i.e. the average of the embeddings of the terms in the document). Use these to cluster documents, back out terms. Repeat process above.
# - [ ] Clean code, add comments and fill docstrings.
# - [ ] Prepare PR.
#
# ### Questions:
# - Clusters with higher n_clust vs. re-clustering within each cluster?
# %%
#
# %%
# cluster_dict[label] = dict(zip(embeddings.index, [int(l) for l in km1.labels_]))


# km1, km1cfg = next(cluster_gens[0])
# label = (
#     km1.__module__ +
#     "_" +
#     "_".join(
#         [get_param_initial(i)+str(j) for i,j in km1cfg.items()]
#     )
# )


# %%
