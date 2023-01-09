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
import boto3, pickle
from IPython.display import display
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from hdbscan import HDBSCAN
from collections import defaultdict
from dap_aria_mapping import PROJECT_DIR, BUCKET_NAME, logger
from dap_aria_mapping.pipeline.semantic_taxonomy.utils import (
    make_dataframe,
    make_cooccurrences,
    run_clustering_generators
)

np.random.seed(42)
s3 = boto3.client("s3")
try:
    try:
        logger.info("Downloading embeddings from S3")
        embeddings_object = s3.get_object(
            Bucket=BUCKET_NAME,
            Key="outputs/embeddings/embeddings.pkl"
        )
        embeddings = pickle.loads(embeddings_object["Body"].read())
    except:
        logger.info("Failed to download from S3. Attempting to load from local instead")
        with open(f"{PROJECT_DIR}/outputs/embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
except:
    logger.info("Failed to load embeddings. Running pipeline with default (test) parameters")
    import subprocess
    subprocess.run(
        f"python {PROJECT_DIR}/dap_aria_mapping/pipeline/embeddings/make_embeddings.py", 
        shell=True
    )
    with open(f"{PROJECT_DIR}/outputs/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

embeddings = pd.DataFrame.from_dict(embeddings).T

embeddings_2d = umap.UMAP(
    n_neighbors=5,
    min_dist=0.05,
    n_components=2
).fit_transform(embeddings)

# Strictly hierarchical clustering
cluster_configs_s = [
    [
        KMeans,
        [
            {"n_clusters": 5, "n_init": 5},  # parent level
            {"n_clusters": 20, "n_init": 5}, # nested level 1
            {"n_clusters": 20, "n_init": 5}, # nested level 2
            {"n_clusters": 40, "n_init": 5}  # nested level 3
        ],
    ],
    [
        AgglomerativeClustering,
        [
            {"n_clusters": 5}, # parent level
            {"n_clusters": 20}, # nested level 1
            {"n_clusters": 20}, # nested level 2
            {"n_clusters": 40}  # nested level 3
        ],
    ],
]
cluster_configs_simb = [
    [
        KMeans,
        [
            {"n_clusters": 5, "n_init": 5}, # parent level
            {"n_clusters": 20, "n_init": 5},# nested level 1, total n_clusters is 20+
            {"n_clusters": 20, "n_init": 5},# nested level 2, total n_clusters is 20+
            {"n_clusters": 40, "n_init": 5},# nested level 2, total n_clusters is 40+
        ],
    ],
]
cluster_configs_f = [
    [KMeans, [{"n_clusters": [5, 20, 20, 40], "n_init": 5}]], # level 1, level 2, level 3, level 4
    [AgglomerativeClustering, [{"n_clusters": [5, 20, 20, 40]}]], # level 1, level 2, level 3, level 4
    [DBSCAN, [{"eps": [0.15, 0.25], "min_samples": [8, 16]}]], # level 1, level 2, level 3, level 4
    [HDBSCAN, [{"min_cluster_size": [4, 8], "min_samples": [8, 16]}]], # level 1, level 2, level 3, level 4
]
cluster_configs_dendrogram = [[AgglomerativeClustering, [{"n_clusters": 100}]]]
cluster_configs_centroids = [
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
# %%
logger.info("Running clustering generators - Strict hierarchical clustering")
# run clustering generators
cluster_outputs_s, _ = run_clustering_generators(
    cluster_configs_s, 
    embeddings
)
logger.info("Running clustering generators - Strict hierarchical clustering with imbalanced nested clusters")
# Strict hierarchical clustering with imbalanced nested clusters
cluster_outputs_simb, _ = run_clustering_generators(
    cluster_configs_simb, 
    embeddings, 
    imbalanced=True
)
logger.info("Running clustering generators - Fuzzy hierarchical clustering")
# Fuzzy hierarchical clustering
cluster_outputs_f_, _ = run_clustering_generators(
    cluster_configs_f, 
    embeddings
)
logger.info("Running clustering generators - Dendrogram")
# Using dendrograms from Agglomerative Clustering (enforces hierarchy)
cluster_outputs_d, _ = run_clustering_generators(
    cluster_configs_dendrogram, 
    embeddings, 
    dendrogram_levels=200
)
logger.info("Running clustering generators - Centroids of Kmeans clustering as children nodes for further clustering")
# Centroids of Kmeans clustering as children nodes for further clustering (à la [job skill taxonomy](https://github.com/nestauk/skills-taxonomy-v2/tree/dev/skills_taxonomy_v2/pipeline/skills_taxonomy))
cluster_outputs_c, _ = run_clustering_generators(
    cluster_configs_centroids, 
    embeddings, 
    embeddings_2d=embeddings_2d
)
# Necessary changes to homogeneise outputs
# [HACK] flip order, should be fixed in run_clustering_generators (should run highest level → lowest level)
for output_dict in cluster_outputs_c:
    for k,v in output_dict["labels"].items():
        output_dict["labels"][k] = v[::-1]
    output_dict["silhouette"] = output_dict["silhouette"][::-1]
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

logger.info("Running clustering generators - Done")

logger.info("Running dictionary & table exports")
# Relevant dictionary outputs
outputs_ls = [
    cluster_outputs_s[3],
    cluster_outputs_s[7],
    cluster_outputs_simb[-1],
    cluster_outputs_f[0],
    cluster_outputs_f[1],
    cluster_outputs_d,
    cluster_outputs_c[-1],
]
df_configs = [
    {"name": "kmeans_strict", "label": "_strict", "cumulative": True},
    {"name": "agglom_strict", "label": "_strict", "cumulative": True},
    {"name": "kmeans_strict_imb", "label": "_strict_imbalanced", "cumulative": True},
    {"name": "kmeans_fuzzy", "label": "_fuzzy", "cumulative": True},
    {"name": "agglom_fuzzy", "label": "_fuzzy", "cumulative": True},
    {"name": "agglom_dendrogram", "label": "", "cumulative": True},
    {"name": "kmeans_centroid", "label": "_centroids", "cumulative": False}
]

# Create list of output dataframes
outputs_df = []
for dict_, df_config in zip(outputs_ls, df_configs):
    s3 = boto3.client("s3")
    name_export = df_config["name"]
    try:
        s3.put_object(
            Body=pickle.dumps(dict_),
            Bucket=BUCKET_NAME,
            Key=f"outputs/semantic_prototypes/dictionary_outputs/{name_export}.pkl"
        )
    except:
        with open(f"{PROJECT_DIR}/outputs/interim/{name_export}.pkl", "wb") as f:
            pickle.dump(dict_, f)
    outputs_df.append(make_dataframe(dict_, df_config["label"], df_config["cumulative"]))

# Silhouette Scores
results = {"_".join([k["name"],str(id)]): e for v,k in list(zip(outputs_ls, df_configs)) for id, e in enumerate(v["silhouette"])}

silhouette_df = pd.DataFrame(results, index=["silhouette"]).T.sort_values(
    "silhouette", ascending=False
)
try:
    silhouette_df.to_parquet(f"s3://aria-mapping/outputs/semantic_prototypes/silhouette_scores.parquet")
except:
    silhouette_df.to_parquet(f"{PROJECT_DIR}/outputs/interim/silhouette_scores.parquet")
display(silhouette_df)

# Meta Clustering
meta_cluster_df = pd.concat(outputs_df, axis=1)
# %%
# Exports co-occurrences and tables of cluster assignments
labels = [x["name"] for x in df_configs]
for label, df in zip(
    labels + ["meta_cluster"],
    outputs_df + [meta_cluster_df]
):
    df_ = df.reset_index().rename(columns={"index": "tag"})
    cooccur_dict = make_cooccurrences(df_)
    cooccur_df = pd.DataFrame(cooccur_dict, index=df_["tag"], columns=df_["tag"])
    try:
        df.to_parquet(f"s3://aria-mapping/outputs/semantic_prototypes/assignments/{label}.parquet")
        cooccur_df.to_parquet(f"s3://aria-mapping/outputs/semantic_prototypes/cooccurrences/{label}.parquet")
    except:
        df.to_parquet(f"{PROJECT_DIR}/outputs/interim/assignments/{label}.parquet")
        cooccur_df.to_parquet(f"{PROJECT_DIR}/outputs/interim/cooccurrences/{label}.parquet")
    
    cooccur_labelled_dict = {
        k: {e: int(v) for e, v in zip(df_["tag"], cooccur_dict[i])}
        for i, k in enumerate(df_["tag"])
    }
    with open(f"{PROJECT_DIR}/outputs/interim/assignments/{label}_dict.pkl", "wb") as f:
        pickle.dump(cooccur_labelled_dict, f)

cooccur_df.head(10)
# %%
