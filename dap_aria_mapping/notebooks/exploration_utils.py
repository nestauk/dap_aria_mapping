import numpy as np
import pandas as pd
import re
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from typing import Union, Dict, Sequence, Callable, Tuple, Generator
from itertools import product
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import chain, count


def get_sample(entities, score_threshold, num_articles=-1):
    if num_articles == -1:
        num_articles = len(entities.values())
    draws = num_articles / len(entities.values()) >= np.random.uniform(
        0, 1, len(entities.values())
    )
    sample = {k: entities[k] for i, k in enumerate(entities.keys()) if draws[i]}
    return {k: [e[0] for e in v if e[1] >= score_threshold] for k, v in sample.items()}


def filter_entities(entities, method="percentile", min_freq=0, max_freq=100):
    assert min_freq < max_freq, "min_freq must be less than max_freq"
    assert method in ["percentile", "absolute"], "method must be in ['percentile', 'absolute']"
    frequencies = Counter(chain(*entities.values()))
    if method == "percentile":
        min_freq, max_freq = (
            np.percentile(list(frequencies.values()), min_freq),
            np.percentile(list(frequencies.values()), max_freq),
        )
    return {
        k: [e for e in v if min_freq <= frequencies[e] <= max_freq]
        for k, v in entities.items()
    }


def embed(entities, model):
    model = SentenceTransformer(model)
    return pd.DataFrame(model.encode(entities), index=entities)


def cluster_to_list(cluster, cdict, embeddings):
    return [e for e in embeddings.index if cdict[e] == cluster]


def get_cluster_config(params):
    if any([isinstance(x, list) for x in params.values()]):
        keys, permuts = (
            list(params.keys()),
            list(
                product(*[v if isinstance(v, list) else [v] for v in params.values()])
            ),
        )
        for perm in permuts:
            yield {k: v for k, v in zip(keys, perm)}
    else:
        yield params


def get_param_initial(query, key):
    if m := re.match(query, key):
        return m.group(2)


def get_manifold_cluster_config(nested_config):
    for child_params in nested_config:
        yield get_cluster_config(child_params)


def update_dictionary(cdict, embeddings, cluster):
    if isinstance(cluster, dict):
        for key in embeddings.index:
            cdict[key].append(cluster[key])
    else:
        for key, date in zip(embeddings.index, cluster.labels_):
            cdict[key].append(date)


# def get_silhouette_score(embeddings, cdict):
#     if re.findall("ce(\d)+", str(list(cdict.values())[0][-1])):
#         sil_dict = {
#             key: re.findall("ce(\d)+", val[-1])[-1]
#             for key, val in cdict.items()
#             if key != 1
#         }
#     else:
#         sil_dict = {
#             key: "".join(map(str, val)) for key, val in cdict.items() if key != 1
#         }
#     if len(set(sil_dict.values())) > 1:
#         return silhouette_score(X=embeddings, labels=list(sil_dict.values()))
#     else:
#         return 0

def get_silhouette_score(embeddings, cdict):
    if isinstance(list(cdict.values())[0], int):
        return compute_silhouette_score(embeddings, cdict)
    elif isinstance(list(cdict.values())[0], list):
        sil_list = []
        for level in range(1, len(list(cdict.values())[0])+1):
            cdict_level = {key: val[:level] for key, val in cdict.items()}
            sil_list.append(compute_silhouette_score(embeddings, cdict_level))
        return sil_list


def compute_silhouette_score(embeddings, cdict):
    if re.findall("ce(\d)+", str(list(cdict.values())[0][-1])):
        sil_dict = {
            key: re.findall("ce(\d)+", val[-1])[-1]
            for key, val in cdict.items()
            if key != 1
        }
    else:
        sil_dict = {
            key: "".join(map(str, val)) for key, val in cdict.items() if key != 1
        }
    if len(set(sil_dict.values())) > 1:
        return silhouette_score(X=embeddings, labels=list(sil_dict.values()))
    else:
        return 0



def clustering_routine(method_class, config, embeddings, **kwargs):
    nested = False
    for child_config in config:
        for param_config in child_config:
            centroids = param_config.pop("centroids", False)

            if nested is False:
                cluster = method_class(**param_config)
                cluster.fit(embeddings)
                cdict = defaultdict(list)
                update_dictionary(cdict, embeddings, cluster)
                yield {
                    "labels": deepcopy(cdict),
                    "model": cluster,
                    "silhouette": get_silhouette_score(embeddings, cdict),
                    "centroid_params": None,
                }

            elif not centroids:
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
                yield {
                    "labels": deepcopy(cdict),
                    "model": cluster,
                    "silhouette": get_silhouette_score(embeddings, cdict),
                    "centroid_params": None,
                }

            elif centroids:
                clusters = list(
                    list(e) for e in set([tuple(i) for i in cdict.values()])
                )
                if not any(["ce" in str(x) for x in clusters[0]]):
                    centdict = defaultdict(list)
                    cent2ddict = defaultdict(list)
                    centmap = defaultdict(list)

                    for clust_group in clusters:
                        centroid_label = "".join(map(str, clust_group))
                        nested_embeddings = embeddings.loc[
                            embeddings.index.isin(
                                [e for e in embeddings.index if cdict[e] == clust_group]
                            )
                        ]
                        embed_indices = [
                            embeddings.index.get_loc(e)
                            for e in nested_embeddings.index
                            if cdict[e] == clust_group
                        ]
                        centdict.update(
                            {f"cl{centroid_label}": nested_embeddings.mean().to_list()}
                        )
                        centmap.update(
                            {f"cl{centroid_label}": nested_embeddings.index.to_list()}
                        )
                        if "embeddings_2d" in kwargs:
                            cent2ddict.update(
                                {
                                    f"cl{centroid_label}": np.mean(
                                        [
                                            kwargs["embeddings_2d"][i]
                                            for i in embed_indices
                                        ],
                                        axis=0,
                                    )
                                }
                            )

                else:
                    clusters = set([clust[-1] for clust in clusters])
                    centdict = defaultdict(list)
                    cent2ddict = defaultdict(list)
                    centmap = defaultdict(list)

                    for clust_group in clusters:
                        nested_embeddings = embeddings.loc[
                            embeddings.index.isin(
                                [
                                    e
                                    for e in embeddings.index
                                    if cdict[e][-1] == clust_group
                                ]
                            )
                        ]
                        embed_indices = [
                            embeddings.index.get_loc(e)
                            for e in nested_embeddings.index
                            if cdict[e][-1] == clust_group
                        ]

                        centroid_label = "".join(map(str, clust_group))

                        centdict.update(
                            {f"{centroid_label}": nested_embeddings.mean().to_list()}
                        )
                        centmap.update(
                            {f"{centroid_label}": nested_embeddings.index.to_list()}
                        )
                        if "embeddings_2d" in kwargs:
                            cent2ddict.update(
                                {
                                    f"{centroid_label}": np.mean(
                                        [
                                            kwargs["embeddings_2d"][i]
                                            for i in embed_indices
                                        ],
                                        axis=0,
                                    )
                                }
                            )
                centroid_embeddings = list(centdict.values())
                sizes = [len(x) for x in centmap.values()]
                try:
                    n_embeddings_2d = np.vstack([ar for ar in cent2ddict.values()])
                except:
                    pass
                cluster = method_class(**param_config)
                cluster.fit(centroid_embeddings)

                centroid_labels = list(zip(centdict.keys(), cluster.labels_))
                cedict = dict(
                    chain(
                        *[
                            list(zip(f := centmap[k], [f"ce{v}"] * len(f)))
                            for k, v in centroid_labels
                        ]
                    )
                )
                update_dictionary(cdict, embeddings, cedict)
                if kwargs.get("embeddings_2d", None) is not None:
                    yield {
                        "labels": deepcopy(cdict),
                        "model": cluster,
                        "silhouette": get_silhouette_score(embeddings, cdict),
                        "centroid_params": {
                            "sizes": sizes,
                            "n_embeddings_2d": n_embeddings_2d,
                        },
                    }
                else:
                    yield {
                        "labels": deepcopy(cdict),
                        "model": cluster,
                        "silhouette": get_silhouette_score(embeddings, cdict),
                        "centroid_params": {"sizes": sizes},
                    }
        nested = True


def cluster_generator(
    cmethod: Tuple[
        Callable, Sequence[Sequence[Tuple[str, Sequence]]]
    ],  # Seq1 for clean, Seq2 for params, Seq3 for fuzzy
    embeddings: pd.DataFrame,
    **kwargs,
) -> Generator:
    """Note this guy now either takes multiple sequences of params (clean hierarchy) or a
    single sequence with a nested sequence of parameters (fuzzy hierarchy)

    Args:
        cmethod (_type_): _description_
        embeddings (_type_): _description_
        nested (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    cclass, cconfig = cmethod
    nested_cconfig = get_manifold_cluster_config(cconfig)
    return clustering_routine(cclass, nested_cconfig, embeddings, **kwargs)


def make_dict_key(model, config):
    return (
        get_param_initial(r"(.*\.)?(.*)", model.__module__)
        + "_"
        + "_".join(
            [get_param_initial(r"(.*_)?(.*)", i) + str(j) for i, j in config.items()]
        )
    )


def get_cluster_outputs(generators):
    cluster_dicts = [
        {
            "labels": clusterdict["labels"],
            "model": clusterdict["model"],
            "silhouette": clusterdict["silhouette"],
            "centroid_params": clusterdict["centroid_params"],
        }
        for ncgen in generators
        for clusterdict in ncgen
    ]
    plot_dicts = [
        [
            {k: "".join(map(str, v)) for k, v in clusters["labels"].items()},
            clusters["model"],
        ]
        for clusters in cluster_dicts
    ]

    if len(cluster_dicts) == 1 or len(plot_dicts) == 1:
        cluster_dicts, plot_dicts = cluster_dicts[0], plot_dicts[0]
    return cluster_dicts, plot_dicts


def run_clustering_generators(cluster_configs, embeddings, build_dendrogram=False, **kwargs):
    cluster_gens = [
        cluster_generator(cluster_config, embeddings, **kwargs)
        for cluster_config in cluster_configs
    ]
    if build_dendrogram:
        cluster_outputs, _ = get_cluster_outputs(cluster_gens)
        dendrogram = make_dendrogram(
            cluster_dict=cluster_outputs["labels"], 
            model=cluster_outputs["model"]
        )
        return (
            make_dendrogram_climb(
                dendrogram=dendrogram, 
                num_levels=6, 
                embeddings=embeddings
            ),
            None
        )
    return get_cluster_outputs(cluster_gens)


def make_subplot_embeddings(embeddings, clabels, axis, label=None, **kwargs):
    axis.scatter(embeddings[:, 0], embeddings[:, 1], c=clabels, **kwargs)
    if label is not None:
        axis.set_title(label)


def make_dendrogram(cluster_dict, model, **kwargs):

    ii = count(len(cluster_dict))
    clusters = [
        {"node_id": next(ii), "left": x[0], "right": x[1]} for x in model.children_
    ]  # Note this works with fuzzy hierarchicals (can't keep tally of tree for strictly hierarchical)
    dendrogram = []
    child_dict = {}

    # populate child dict
    for li in clusters:
        if child_dict.get((node := li["node_id"])) is None:
            child_dict[node] = set([li["right"], li["left"]])
    dendrogram.append(child_dict)

    while len(child_dict) > 1:  # Continue until only one node remains
        # Create a copy of the current-level children nodes
        tmp_dict, parent_dict = deepcopy(child_dict), dict()
        # Iterate over current-level children nodes
        for child_key, _ in sorted(child_dict.items(), key=lambda x: x[0]):
            # If the current child is contained within another node
            if any(child_of := [child_key in v for v in child_dict.values()]):
                # Identify the parent node
                parent_key = list(child_dict.keys())[np.where(child_of)[0][0]]
                try:  # try clause in case child key is no longer available in child dict
                    # if this is the lowest level of the dendrogram, ie no further downstream nodes
                    if all(
                        [k not in list(tmp_dict.keys()) for k in tmp_dict[child_key]]
                    ):  # on if condition yielding error, it implies tmp_dict does not contain child_key and jumps to passing parent through (if applicable) (some child of current node already updated and popped)
                        try:
                            tmp_dict[parent_key].update(tmp_dict[child_key])
                            tmp_dict[parent_key].remove(child_key)
                            parent_dict[parent_key] = tmp_dict.pop(parent_key)
                            # tmp_dict.pop(child_key, None)
                        except:  # If it has already been popped, update directly (because second child still sat at bottom of dendrogram)
                            parent_dict[parent_key].update(tmp_dict[child_key])
                            parent_dict[parent_key].remove(child_key)
                            # tmp_dict.pop(child_key, None)
                except:  # passes through "idle" parent node (as children were not at the bottom of dendrogram)
                    if parent_key not in parent_dict.keys():
                        parent_dict[parent_key] = tmp_dict.pop(parent_key)
        # Parent dict is the new child dict
        child_dict = deepcopy(parent_dict)
        # Add the new child dict to the dendrogram
        dendrogram.append(child_dict)
    return dendrogram

def climb_step(dendrogram, level, embeddings):
    split = dendrogram[-(level+2)]
    tags = [
        [tag for tag in cluster_tags if tag < embeddings.shape[0]]
        for cluster_tags in split.values()
    ]
    tags = [tag for tag in tags if len(tag) > 0]
    level_clust = list(
        chain(
            *[
                list(zip(tags[cluster], [cluster] * len(tags[cluster])))
                for cluster in range(len(tags))
            ]
        )
    )
    level_clust = sorted(level_clust, key=lambda x: x[0]) #guarantees tag order consistency
    return dict(zip(embeddings.index, [x[1] for x in level_clust]))

def make_dendrogram_climb(dendrogram, num_levels, embeddings):
    cdict = defaultdict(list)
    for level in range(num_levels):
        level_clust = climb_step(dendrogram, level, embeddings)
        for key, value in level_clust.items():
            cdict[key].append(value)
    return {
        "labels": cdict,
        "model": "dendrogram",
        "silhouette": get_silhouette_score(embeddings=embeddings, cdict=cdict),
        "centroid_params": None
    }