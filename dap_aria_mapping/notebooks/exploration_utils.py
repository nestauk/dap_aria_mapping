import numpy as np
import pandas as pd
import re, sklearn
import matplotlib.axes as mpl_axes
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from typing import Union, Dict, DefaultDict, List, Any, Set, Type, Literal, Sequence, Callable, Tuple, Generator
from itertools import product
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import chain, count
import altair as alt


def get_sample(
    entities: Dict[str, Sequence[Tuple[str, int]]], 
    score_threshold: int = 80, 
    num_articles: int = -1
) -> Dict[str, Sequence[str]]:
    """Creates a sample of entities from the entities dictionary,
    where each article contains a list of tuples of the form (entity, score).
    The sample size is determined by `num_articles`, and entities with 
    a confidence score below the threshold are removed.

    Args:
        entities (Dict[str, Sequence[Tuple[str, int]]]): A dictionary of entities, 
            where each key is an article and each value is a list of tuples of the 
            form (entity, score).
        score_threshold (int, optional): The minimum confidence score for an entity.
            If not provided, defaults to 80.
        num_articles (int, optional): The number of articles to sample. If -1,
            all articles are sampled. If not provided, defaults to -1.

    Returns:
        Dict[str, Sequence[str]]: A dictionary of entities, where each key is 
            an article and each value is a list of entities.
    """    
    if num_articles == -1:
        num_articles = len(entities.values())
    draws = num_articles / len(entities.values()) >= np.random.uniform(
        0, 1, len(entities.values())
    )
    sample = {k: entities[k] for i, k in enumerate(entities.keys()) if draws[i]}
    return {k: [e[0] for e in v if e[1] >= score_threshold] for k, v in sample.items()}


def filter_entities(
    entities: Dict[str, Sequence[str]], 
    method: Literal["percentile", "absolute"] = "percentile", 
    min_freq: int = 0, 
    max_freq: int = 100
) -> Dict[str, Sequence[str]]:
    """Filters entities from the entities dictionary, where each article contains 
    a list of entities. The filtering is done by frequency, either as a percentile
    or as an absolute value.

    Args:
        entities (Dict[str, Sequence[str]]): A dictionary of entities, where each
            key is an article and each value is a list of entities.
        method (Literal[percentile, absolute], optional): The method of filtering.
            Defaults to "percentile".
        min_freq (int, optional): The minimum frequency for an entity. If `method`
            is "percentile", this is the minimum percentile. If `method` is "absolute",
            this is the minimum absolute frequency. Defaults to 0.
        max_freq (int, optional): The maximum frequency for an entity. If `method`
            is "percentile", this is the maximum percentile. If `method` is "absolute",
            this is the maximum absolute frequency. Defaults to 100.

    Returns:
        Dict[str, Sequence[str]]: A dictionary of entities, where each key is
            an article and each value is a list of entities.
    """    
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


def embed(
    entities: Sequence[str],
    model: str = "distilbert-base-nli-stsb-mean-tokens",
) -> pd.DataFrame:
    """Embeds a list of entities using a pretrained sentence transformer model.

    Args:
        entities (Sequence[str]): A list of entities.
        model (str, optional): The name of the pretrained model. Defaults to 
            "distilbert-base-nli-stsb-mean-tokens".

    Returns:
        pd.DataFrame: A dataframe of embeddings, where each row is an entity
            and each column is a dimension of the embedding.
    """
    model = SentenceTransformer(model)
    return pd.DataFrame(model.encode(entities), index=entities)


def cluster_to_list(
    cluster: Union[int, str],
    cdict: Dict[str, Union[int, str]],
    embeddings: pd.DataFrame,
) -> List[str]:
    """Maps from a cluster assignment to a list of relevant entities.

    Args:
        cluster (Union[int, str]): An assignment to a cluster for a given method.
        cdict (Dict[str, Union[int, str]]): A dictionary mapping entities to 
            cluster assignments.
        embeddings (pd.DataFrame): A dataframe of embeddings, where each row is
            an entity and each column is a dimension of the embedding.

    Returns:
        List[str]: A list of entities that are assigned to the given cluster.
    """
    return [e for e in embeddings.index if cdict[e] == cluster]


def get_cluster_config(
    params: Dict[str, Union[int, str, List[Union[int, str]]]]
) -> Generator[Dict[str, Union[int, str]], None, None]:
    """Generates a list of dictionaries of parameters for clustering. If any
    parameter is a list, then the cartesian product of the parameters is taken.

    Args:
        params (Dict[str, Union[int, str, List[Union[int, str]]]]): A dictionary
            of parameters for clustering.

    Yields:
        Generator[Dict[str, Union[int, str]], None, None]: A generator of dictionaries.
    """    
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


def get_param_initial(query: str, key: str) -> str:
    """Gets the initial value of a parameter from a key.

    Args:
        query (str): Regex query.
        key (str): Key.

    Returns:
        str: Initial value of the parameter.
    """    
    if m := re.match(query, key):
        return m.group(2)


def get_manifold_cluster_config(
    nested_config: List[Dict[str, Union[int, str]]]
) -> Generator[Dict[str, Union[int, str]], None, None]:
    """Generates a list of dictionaries of parameters for manifold clustering. If 
    any nested dictionary contains parameters with lists, then the cartesian product 
    of these parameters is taken. The latter creates a generator with nested generators.

    Args:
        nested_config (List[Dict[str, Union[int, str]]]): A list of dictionaries.

    Yields:
        Generator[Dict[str, Union[int, str]], None, None]: A generator of nested dictionaries.
    """
    for child_params in nested_config:
        yield get_cluster_config(child_params)


def update_dictionary(
    cdict: DefaultDict[str, Union[Sequence[Union[int, str]], Union[int, str]]], 
    embeddings: pd.DataFrame, 
    cluster: sklearn.base.ClusterMixin
) -> DefaultDict[str, Union[Sequence[Union[int, str]], Union[int, str]]]:
    """Updates a dictionary of cluster assignments.

    Args:
        cdict (DefaultDict[str, Union[Sequence[Union[int, str]], Union[int, str]]]):
            The child dictionary of cluster assignments.
        embeddings (pd.DataFrame): A dataframe of embeddings, where each row is
            an entity and each column is a dimension of the embedding.
        cluster (sklearn.base.ClusterMixin): An sklearn cluster object.

    Returns:
        DefaultDict[str, Union[Sequence[Union[int, str]], Union[int, str]]]:
            The updated dictionary of cluster assignments.
    """
    if isinstance(cluster, dict):
        for key in embeddings.index:
            cdict[key].append(cluster[key])
    else:
        for key, date in zip(embeddings.index, cluster.labels_):
            cdict[key].append(date)


def get_silhouette_score(
    embeddings: pd.DataFrame, 
    cdict: Dict[str, Union[Sequence[Union[int, str]], Union[int, str]]]
) -> Union[float, List[float]]:
    """Iterates calculations of the silhouette score for a given dictionary 
    of cluster assignments. If the dictionary contains multiple levels of 
    cluster assignments, then the silhouette score is computed for each level.

    Args:
        embeddings (pd.DataFrame): A dataframe of embeddings, where each row is
            an entity and each column is a dimension of the embedding.
        cdict (Dict[str, Union[Sequence[Union[int, str]], Union[int, str]]]): 
            A dictionary of cluster assignments.

    Returns:
        Union[float, List[float]]: The silhouette score for each hierarchical
            level.
    """
    if isinstance(list(cdict.values())[0], int):
        return compute_silhouette_score(embeddings, cdict)
    elif isinstance(list(cdict.values())[0], list):
        sil_list = []
        for level in range(1, len(list(cdict.values())[0])+1):
            cdict_level = {key: val[:level] for key, val in cdict.items()}
            sil_list.append(compute_silhouette_score(embeddings, cdict_level))
        return sil_list


def compute_silhouette_score(
    embeddings: pd.DataFrame, 
    cdict: Dict[str, Union[Sequence[Union[int, str]], Union[int, str]]]
) -> float:
    """Calculates the silhouette score for a given dictionary of cluster assignments.
    If the dictionary contains centroid clusters, then the silhouette score is computed
    for each centroid cluster. Otherwise, the silhouette score is computed for the
    entity-specific sequences of cluster assignments.

    Args:
        embeddings (pd.DataFrame): A dataframe of embeddings, where each row is
            an entity and each column is a dimension of the embedding.
        cdict (Dict[str, Union[Sequence[Union[int, str]], Union[int, str]]]):
            A dictionary of cluster assignments.

    Returns:
        float: The silhouette score.
    """
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

class ClusteringRoutine(object):
    """A class for clustering routines. This class is used to cluster embeddings
    using a given clustering method and a list of parameter configurations.
    """
    def __init__(self, embeddings: pd.DataFrame):
        """Initializes a clustering routine. 

        Args:
            embeddings (pd.DataFrame): A dataframe of embeddings, where each row is
                an entity and each column is a dimension of the embedding.
        """        
        self.embeddings = embeddings

    def make_init_iteration(
        self, 
        param_config: Dict[str, Union[int, str]]
    ) -> Generator[Dict[str, Union[int, str]], None, None]:
        """Makes the top-level iteration of a clustering routine. Creates a cluster
        object and fits it to the embeddings. Creates and updates the dictionary of 
        cluster assignments. Yields the dictionary of cluster assignments, the cluster
        object, the silhouette score, and the centroid parameters.

        Args:
            param_config (Dict[str, Union[int, str]]): A dictionary of parameter
                configurations.

        Yields:
            Generator[Dict[str, Union[int, str]], None, None]: A generator that yields
                the dictionary of cluster assignments, the cluster object, the silhouette
                score, and the centroid parameters (if applies).
        """        
        cluster = self.method_class(**param_config)
        cluster.fit(self.embeddings)
        self.cdict, self.mlist = defaultdict(list), []
        update_dictionary(self.cdict, self.embeddings, cluster), self.mlist.append(cluster)
        yield {
            "labels": deepcopy(self.cdict),
            "model": deepcopy(self.mlist),
            "silhouette": get_silhouette_score(self.embeddings, self.cdict),
            "centroid_params": None,
        }

    def make_iteration(
        self, 
        param_config: Dict[str, Union[int, str]]
    ) -> Generator[Dict[str, Union[int, str]], None, None]:
        """Makes a further iteration of a clustering routine. Creates a cluster object and 
        fits it to the embeddings. Creates and updates the dictionary of cluster assignments.
        Yields the dictionary of cluster assignments and the cluster object, the silhouette
        score, and the centroid parameters.

        Args:
            param_config (Dict[str, Union[int, str]]): A dictionary of parameter
                configurations.

        Yields:
            Generator[Dict[str, Union[int, str]], None, None]: A generator that yields
                the dictionary of cluster assignments, the cluster object, the silhouette
                score, and the centroid parameters (if applies).
        """        
        clusters = list(
            list(e) for e in set([tuple(i) for i in self.cdict.values()])
        )
        for cluster_group in clusters:
            nested_embeddings = self.embeddings.loc[
                self.embeddings.index.isin(
                    [e for e in self.embeddings.index if self.cdict[e] == cluster_group]
                )
            ]
            if all([self.kwargs.get("imbalanced", False), param_config.get("n_clusters", False)]):
                param_config_imb = deepcopy(param_config)
                param_config_imb["n_clusters"] = int(
                    round(
                        nested_embeddings.shape[0] / self.embeddings.shape[0]
                        * param_config["n_clusters"]
                    )
                )
                if param_config_imb["n_clusters"]<2:
                    param_config_imb["n_clusters"] = 2
                cluster = self.method_class(**param_config_imb)
            else:
                cluster = self.method_class(**param_config)
            cluster.fit(nested_embeddings)
            update_dictionary(self.cdict, nested_embeddings, cluster)
        self.mlist.append(cluster)
        yield {
            "labels": deepcopy(self.cdict),
            "model": deepcopy(self.mlist),
            "silhouette": get_silhouette_score(self.embeddings, self.cdict),
            "centroid_params": None,
        }

    def make_centroid_iteration(
        self, 
        param_config: Dict[str, Union[int, str]]
    ) -> Generator[Dict[str, Union[int, str]], None, None]:
        """Makes an iteration of a centroid clustering routine. Creates a cluster object
        and fits it to the embeddings. Creates and updates several dictionaries of cluster
        assignments, centroid embeddings, and a mapping of centroid to individual tags. 
        Yields the dictionary of cluster assignments, the cluster object, the silhouette
        score, and the centroid parameters.

        Args:
            param_config (Dict[str, Union[int, str]]): A dictionary of parameter
                configurations.

        Yields:
            Generator[Dict[str, Union[int, str]], None, None]: A generator that yields
                the dictionary of cluster assignments, the cluster object, the silhouette
                score, and the centroid parameters (if applies).
        """        
        self.centdict = defaultdict(list)
        self.cent2ddict = defaultdict(list)
        self.centmap = defaultdict(list)
        clusters = list(
            list(e) for e in set([tuple(i) for i in self.cdict.values()])
        )
        if not any(["ce" in str(x) for x in clusters[0]]):
            self.make_centroid_init_step(clusters)
        else:
            self.make_centroid_child_step(clusters)
        centroid_embeddings = list(self.centdict.values())
        sizes = [len(x) for x in self.centmap.values()]
        try:
            n_embeddings_2d = np.vstack([ar for ar in self.cent2ddict.values()])
        except:
            pass
        cluster = self.method_class(**param_config)
        cluster.fit(centroid_embeddings)

        centroid_labels = list(zip(self.centdict.keys(), cluster.labels_))
        cedict = dict(
            chain(
                *[
                    list(zip(f := self.centmap[k], [f"ce{v}"] * len(f)))
                    for k, v in centroid_labels
                ]
            )
        )
        update_dictionary(self.cdict, self.embeddings, cedict), self.mlist.append(cluster)
        if self.kwargs.get("embeddings_2d", None) is not None:
            yield {
                "labels": deepcopy(self.cdict),
                "model": deepcopy(self.mlist),
                "silhouette": get_silhouette_score(self.embeddings, self.cdict),
                "centroid_params": {
                    "sizes": sizes,
                    "n_embeddings_2d": n_embeddings_2d,
                },
            }
        else:
            yield {
                "labels": deepcopy(self.cdict),
                "model": deepcopy(self.mlist),
                "silhouette": get_silhouette_score(self.embeddings, self.cdict),
                "centroid_params": {"sizes": sizes},
            }

    def make_centroid_init_step(self, clusters: List[List[str]]) -> None:
        """Creates the initial centroid step of a centroid clustering routine. Updates
        several dictionaries of cluster assignments, centroid embeddings, and a mapping of
        centroid to individual tags.

        Args:
            clusters (List[List[str]]): A list of lists of cluster assignments.
        """        
        for clust_group in clusters:
            centroid_label = "".join(map(str, clust_group))
            nested_embeddings = self.embeddings.loc[
                self.embeddings.index.isin(
                    [e for e in self.embeddings.index if self.cdict[e] == clust_group]
                )
            ]
            embed_indices = [
                self.embeddings.index.get_loc(e)
                for e in nested_embeddings.index
                if self.cdict[e] == clust_group
            ]
            self.centdict.update(
                {f"cl{centroid_label}": nested_embeddings.mean().to_list()}
            )
            self.centmap.update(
                {f"cl{centroid_label}": nested_embeddings.index.to_list()}
            )
            if "embeddings_2d" in self.kwargs:
                self.cent2ddict.update(
                    {
                        f"cl{centroid_label}": np.mean(
                            [
                                self.kwargs["embeddings_2d"][i]
                                for i in embed_indices
                            ],
                            axis=0,
                        )
                    }
                )
    
    def make_centroid_child_step(self, clusters: List[List[str]]) -> None:
        """Creates the child centroid step of a centroid clustering routine. Updates
        several dictionaries of cluster assignments, centroid embeddings, and a mapping of
        centroid to individual tags.

        Args:
            clusters (List[List[str]]): A list of lists of cluster assignments.
        """        
        clusters = set([clust[-1] for clust in clusters])
        for clust_group in clusters:
            centroid_label = "".join(map(str, clust_group))
            nested_embeddings = self.embeddings.loc[
                self.embeddings.index.isin(
                    [
                        e
                        for e in self.embeddings.index
                        if self.cdict[e][-1] == clust_group # [TODO]
                    ]
                )
            ]
            embed_indices = [
                self.embeddings.index.get_loc(e)
                for e in nested_embeddings.index
                if self.cdict[e][-1] == clust_group # [TODO]
            ]

            self.centdict.update(
                {f"{centroid_label}": nested_embeddings.mean().to_list()}
            )
            self.centmap.update(
                {f"{centroid_label}": nested_embeddings.index.to_list()}
            )
            if "embeddings_2d" in self.kwargs:
                self.cent2ddict.update(
                    {
                        f"{centroid_label}": np.mean(
                            [
                                self.kwargs["embeddings_2d"][i]
                                for i in embed_indices
                            ],
                            axis=0,
                        )
                    }
                )
    
    def run_step(self, params_config: Dict[str, Any]) -> Generator:
        """Runs a single step of the clustering routine. This is a generator function
        that yields a dictionary of cluster assignments, a list of clustering models, and
        the silhouette score of the clustering routine.

        Args:
            params_config (Dict[str, Any]): A dictionary of parameters for the clustering
                routine.

        Yields:
            Generator: A generator that yields a dictionary of cluster assignments, a list
                of clustering models, and the silhouette score of the clustering routine.
        """        
        centroids = params_config.pop("centroids", False)
        if self.nested is False:
            yield from self.make_init_iteration(params_config)
            # self.nested = True
        else:
            if centroids:
                yield from self.make_centroid_iteration(params_config)
            else:
                yield from self.make_iteration(params_config)

    def run_single_routine(
        self, 
        method_class: Type[sklearn.base.ClusterMixin], 
        config: Dict[str, Any],
        **kwargs
    ) -> Generator:
        """Runs a single clustering routine. This is a generator function that yields a
        dictionary of cluster assignments, a list of clustering models, and the silhouette
        score of the clustering routine.

        Args:
            method_class (Type[BaseClusterMethod]): A sklearn class of a clustering method.
            config (Dict[str, Any]): A dictionary of parameters for the clustering routine.

        Returns:
            Generator: A generator that yields a dictionary of cluster assignments, a list
                of clustering models, and the silhouette score of the clustering routine.
        """    
        self.method_class = method_class
        self.nested = False
        self.kwargs = kwargs
        yield from self.run_step(config)

    def run_from_configs(
        self, 
        config: Tuple[Type[sklearn.base.ClusterMixin], Dict[str, Any]],
        **kwargs
    ) -> Generator[Dict[str, Union[List[Union[int, str]], Union[int, str]]], None, None]:
        """Runs a clustering routine from a list of parameters. Lists of parameter dicts
        are interpreted as nested clustering routines, where each child routine is run
        on the subsets generated by the parent routine. If the dictionary parameters contain
        lists, then the routine is run on each combination of parameters. This is a
        generator function that yields a dictionary of cluster assignments, a list of
        clustering models, and the silhouette score of the clustering routine.

        Args:
            config (Tuple[Type[sklearn.base.ClusterMixin], Dict[str, Any]]): A tuple of a
                sklearn class of a clustering method and a dictionary of parameters for the
                clustering routine.

        Yields:
            Generator[Dict[str, Union[List[Union[int, str], Union[int, str]]], None, None]:
                A generator that yields a dictionary of cluster assignments, a list of
                clustering models, and the silhouette score of the clustering routine.
        """    
        self.method_class, cconfig = config
        self.kwargs = kwargs
        self.nested = False
        nested_cconfig = get_manifold_cluster_config(cconfig)

        for child_config in nested_cconfig:
            for param_config in child_config:
                yield from self.run_step(param_config)
            self.nested = True
                
    def __iter__(self):
        return self

def get_cluster_outputs(generators: List[Generator]) -> Tuple[Dict[str, Any], List[Any]]:
    """Gets the outputs of a list of clustering generators.

    Args:
        generators (List[Generator]): A list of clustering generators.

    Returns:
        Tuple[Dict[str, Any], List[Any]]: A tuple of a dictionary of cluster assignments,
            a list of clustering models, and the silhouette score of the clustering
            routine.
    """    
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

def run_clustering_generators(
    cluster_configs: List[Tuple[Type[sklearn.base.ClusterMixin], Dict[str, Any]]],
    embeddings: pd.DataFrame, 
    dendrogram_levels: int = -1, 
    **kwargs
) -> Tuple[Dict[str, Any], List[Any]]:
    """Runs a list of clustering routines. Each routine is a generator that yields a
    dictionary of cluster assignments, a list of clustering models, and the silhouette
    score of the clustering routine. If the dendrogram_levels parameter is set to a value
    other than -1, then a dendrogram is generated from the cluster assignments of the
    last routine in the list.

    Args:
        cluster_configs (List[Tuple[Type[sklearn.base.ClusterMixin], Dict[str, Any]]]): A
            list of tuples of a sklearn class of a clustering method and a dictionary of
            parameters for the clustering routine.
        build_dendrogram (int, optional): The number of levels to climb the dendrogram. If
            set to -1, then no dendrogram is generated. Defaults to -1.

    Returns:
        Tuple[Dict[str, Any], List[Any]]: A tuple of a dictionary of cluster assignments,
            a list of clustering models, and the silhouette score of the clustering
            routine.
    """
    cluster_gens = []
    for cluster_config in cluster_configs:
        routine = ClusteringRoutine(embeddings)
        cluster_gens.append(routine.run_from_configs(cluster_config, **kwargs))
    if dendrogram_levels != -1:
        cluster_outputs, _ = get_cluster_outputs(cluster_gens)
        dendrogram = make_dendrogram(
            cluster_dict=cluster_outputs["labels"], 
            model=cluster_outputs["model"][-1]
        )
        return (
            make_dendrogram_climb(
                dendrogram=dendrogram, 
                num_levels=dendrogram_levels, 
                embeddings=embeddings
            ),
            None
        )
    return get_cluster_outputs(cluster_gens)


def make_subplot_embeddings(
    embeddings: pd.DataFrame, 
    clabels: pd.Series,
    axis: mpl_axes.Axes, 
    label: str = None, 
    **kwargs
) -> None:
    """Makes a scatter plot of embeddings with cluster labels.

    Args:
        embeddings (pd.DataFrame): A dataframe of embeddings.
        clabels (pd.Series): A series of cluster labels.
        axis (Callable): An axis object.
        label (str, optional): A subplot title. Defaults to None.
    """
    axis.scatter(embeddings[:, 0], embeddings[:, 1], c=clabels, **kwargs)
    if label is not None:
        axis.set_title(label)


def make_dendrogram(
    cluster_dict: Dict[str, Any],
    model: sklearn.base.ClusterMixin,
) -> List[Dict[int, Set[int]]]:
    """Makes a dendrogram from a cluster dictionary and a clustering model.

    Args:
        cluster_dict (Dict[str, Any]): A dictionary of cluster assignments.
        model (sklearn.base.ClusterMixin): A clustering model, with a tree-based
            structure. It is assumed that the model has a children_ attribute, such
            as with AgglomerativeClustering.

    Returns:
        List[Dict[int, Set[int]]]: A list of dictionaries of children nodes. Each
            dictionary is a level of the dendrogram.
    """

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

def climb_step(
    dendrogram: List[Dict[int, Set[int]]],
    level: int, 
    embeddings: pd.DataFrame
) -> Dict[str, int]:
    """Climbs a dendrogram to a given level and returns a dictionary of cluster
    assignments.

    Args:
        dendrogram (_type_): A list of dictionaries of children nodes. Each
            dictionary is a level of the dendrogram.
        level (int): An integer representing the level of the dendrogram to climb.
        embeddings (pd.DataFrame): A dataframe of embeddings.

    Returns:
        Dict[str, int]: A dictionary of cluster assignments.
    """
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

def make_dendrogram_climb(
    dendrogram: List[Dict[int, Set[int]]],
    num_levels: int, 
    embeddings: pd.DataFrame
) -> Dict[str, Any]:
    """Climbs a dendrogram to a given level and returns a dictionary of cluster
    assignments.

    Args:
        dendrogram (List[Dict[int, Set[int]]]): A list of dictionaries of children 
            nodes. Each dictionary is a level of the dendrogram.
        num_levels (int): An integer representing the number of levels to climb.
        embeddings (pd.DataFrame): A dataframe of embeddings.

    Returns:
        Dict[str, Any]: A dictionary of cluster assignments.
    """
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

def make_plots(analysis_df: pd.DataFrame) -> alt.Chart:
    """Creates a plot from the cluster analysis dataframe. The plot is a
    horizontal concatenation of bar charts, one for each column in the
    dataframe. The bar charts are sorted by the number of observations in each
    cluster. The plot is returned as an altair chart object.

    Args:
        analysis_df (pd.DataFrame): A dataframe of cluster analysis results.

    Returns:
        alt.Chart: A horizontal concatenation of bar charts, one for each column
            in the dataframe.
    """    
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

def make_dataframe(
    cluster_outputs: Dict[str, Any],
    label: str = "cluster", 
    cumulative:bool = True
) -> pd.DataFrame:
    """Creates a dataframe of cluster assignments from the output of a clustering
    algorithm. The dataframe has a column for each level of the clustering
    algorithm, while `label` can be used to annotate the name of the column 
    (e.g. "cluster" or "cluster_level").

    Args:
        cluster_outputs (Dict[str, Any]): A dictionary of cluster assignments.
        label (str, optional): A string to use as the column name. Defaults to
            "cluster".
        cumulative (bool, optional): A boolean indicating whether to use the
            cumulative cluster assignments. Defaults to True, False is intended
            to be used with centroid clustering only.

    Returns:
        pd.DataFrame: A dataframe of cluster assignments.
    """
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
    try:
        model_name = cluster_outputs["model"].__module__.replace(".", "")
    except:
        model_name = cluster_outputs["model"]
    if isinstance(model_name, list):
        columns = [
            "_".join(x) 
            for x in list(
                zip(
                    [str(x) for x in model_name],
                    [str(idx) + label for idx in range(num_levels)]
                )
            )
        ]
    else:
        columns = [
            "{}_{}{}".format(
                model_name, idx, label
            )
            for idx in range(num_levels)
        ]
    return pd.DataFrame.from_dict(
        d, 
        orient="index", 
        columns=columns
    )

def make_cooccurrences(dataframe: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Creates a dictionary of co-occurrence counts from a dataframe of cluster
    assignments. The dictionary is structured as follows:
        {
            term1: {term1: count, term2: count, ...}, 
            term2: {term1: count, term2: count, ...}, 
            ...
        }

    Args:
        dataframe (pd.DataFrame): A dataframe of cluster assignments.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary of co-occurrence counts.
    """    
    assert any(dataframe.columns.isin(["tag"])), "Dataframe must have a tag column"
    dfnp = dataframe[[x for x in dataframe.columns if x != "tag"]].to_numpy()
    m1 = np.tile(dfnp.T, (1, dfnp.shape[0])).flatten()
    m2 = np.repeat(dfnp.T, dfnp.shape[0])
    mf = np.where(m1 == m2, 1, 0)
    cooccur_dict = np.sum(mf.reshape((dfnp.shape[1], dfnp.shape[0], dfnp.shape[0])), axis=0)
    return {
        k: {e: int(v) for e, v in zip(dataframe["tag"], cooccur_dict[i])}
        for i, k in enumerate(dataframe["tag"])
    }
