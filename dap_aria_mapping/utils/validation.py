import numpy as np
import pandas as pd
from typing import (
    Union,
    Dict,
    List,
    Sequence,
    Tuple,
)
import random
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import chain
from toolz import pipe
from dap_aria_mapping import logger, PROJECT_DIR


def get_article_assignments(
    taxonomy: pd.DataFrame,
    entities: Dict[str, Sequence[str]],
    level: Union[str, int] = 1,
) -> Dict[str, Dict[str, str]]:
    """Creates a dictionary of journal IDs to entities and the
    corresponding cluster labels.

    Args:
        taxonomy (pd.DataFrame): A dataframe with hierarchical
            cluster assignments.
        entities (Dict[str, Sequence[str]]): A dictionary of
            OpenAlex article IDs to entities.
        level (str, optional): The level of the taxonomy to use.
            Defaults to 1.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary of article IDs to
            entities and the corresponding cluster labels.

        {
            article_label: {
                entity: label,
                entity: label,
                ...
            }
            ...
        }
    """
    taxonomy_dict = taxonomy["Level_{}".format(str(level))].to_dict()
    oa_assignments = {}
    for k, v in entities.items():
        oa_assignments[k] = {}
        for entity in v:
            if entity in taxonomy_dict:
                oa_assignments[k][entity] = taxonomy_dict[entity]
    return oa_assignments


def get_article_assignments_counts(
    article_assignments: Dict[str, Dict[str, str]]
) -> Dict[str, Counter]:
    """Creates a dictionary of article IDs to the counts of
    cluster labels.

    Args:
        article_assignments (Dict[str, Dict[str, str]]): A dictionary
            of article IDs to entities and the corresponding cluster
            labels.

    Returns:
        Dict[str, Counter]: A dictionary of article IDs to the counts
            of cluster labels.

        {
            article_label: {
                Counter[label: count, label: count, ...]
            }
            ...
        }
    """
    article_assignments_counts = {}
    for k, v in article_assignments.items():
        article_assignments_counts[k] = Counter(v.values())
    return article_assignments_counts


def get_journal_entities(
    oa_works: pd.DataFrame,
    oa_entities: Dict[str, Sequence[str]],
) -> Dict[str, Sequence[str]]:
    """Creates a dictionary of journal IDs to entities.

    Args:
        oa_works (pd.DataFrame): A dataframe of OpenAlex works.
        oa_entities (Dict[str, Sequence[str]]): A dictionary of
            OpenAlex article IDs to entities.

    Returns:
        Dict[str, Sequence[str]]: A dictionary of journal IDs to
            entities and the corresponding cluster labels.

        {
            journal_label: [entity, entity, ...],
            ...
        }

    """
    article_to_journal = (
        oa_works[["work_id", "venue_display_name"]]
        .set_index("work_id")
        .dropna()
        .to_dict()["venue_display_name"]
    )
    journal_entities = defaultdict(list)
    for k, v in oa_entities.items():
        if k in article_to_journal:
            journal_entities[article_to_journal[k]].extend(v)
    return journal_entities


def get_entity_counts(journal_entities: Dict[str, Sequence[str]]) -> Dict[str, int]:
    """Creates a dictionary of entities to counts.

    Args:
        journal_entities (Dict[str, Sequence[str]]): A dictionary of
            journal IDs to entities.

    Returns:
        Dict[str, int]: A dictionary of entities to counts.

        {
            entity: count,
            ...
        }
    """
    return Counter(list(chain(*journal_entities.values())))


def get_level_entity_counts(
    journal_entities: Dict[str, Sequence[str]],
    taxonomy: pd.DataFrame,
    level: Union[str, int] = 1,
) -> Dict[str, Dict[str, int]]:
    """Creates a dictionary of entity counts at a specified level
        of the hierarchical taxonomy.

    Args:
        journal_entities (Dict[str, Sequence[str]]): A dictionary of
            journal IDs to entities.
        taxonomy (pd.DataFrame): A dataframe with hierarchical
            cluster assignments.
        level (str, optional): The level of the taxonomy to use.
            Defaults to 1.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary of cluster labels to
            entity counts.

        {
            cluster_label: {
                entity: count,
                entity: count,
                ...
            },
            ...
        }
    """
    entity_counts = get_entity_counts(journal_entities)

    clust_entities = {}
    for clust in taxonomy["Level_{}".format(str(level))].unique():
        ls_entities = taxonomy.loc[
            taxonomy["Level_{}".format(str(level))] == clust
        ].index.to_list()
        clust_entities[clust] = list(
            filter(lambda x: x in entity_counts.keys(), ls_entities)
        )

    clust_entity_counts = {}
    for k in clust_entities.keys():
        clust_entity_counts[k] = {
            entity: entity_counts[entity] for entity in clust_entities[k]
        }

    return clust_entity_counts


def get_journal_entity_counts(
    journal_entities: Dict[str, Sequence[str]], output: str = "absolute"
) -> Dict[str, Dict[str, int]]:
    """Creates a dictionary of journal IDs to entity counts.

    Args:
        journal_entities (Dict[str, Sequence[str]]): A dictionary of
            journal IDs to entities.
        output (str, optional): Output type, either "absolute" or
            "relative". Defaults to "absolute".

    Returns:
        Dict[str, Dict[str, int]]: A dictionary of journal IDs to
            entity counts.

        {
            journal_label: {
                entity: count,
                entity, count,
                ...
            },
            ...
        }
    """
    assert output in [
        "absolute",
        "relative",
    ], "output must be one of 'absolute' or 'relative'"
    journal_entity_counts = {}
    for k, v in journal_entities.items():
        count = pipe(v, Counter, dict)
        if output == "relative":
            nconst = sum(count.values())
            count.update((k, v / nconst) for k, v in count.items())
        journal_entity_counts[k] = count
    return journal_entity_counts


def get_cluster_entity_labels(
    cluster_counts: Dict[str, List[Tuple[str, int]]], num_entities: int = 10
) -> Dict[str, str]:
    """Creates a dictionary of cluster labels to entities.

    Args:
        cluster_counts (Dict[str, List[Tuple[str, int]]]): A dictionary
            of cluster labels to entity counts.
        num_entities (int, optional): Number of entities to include
            in the label. Defaults to 10.

    Returns:
        Dict[str, str]: A dictionary of cluster labels to entities.

        {
            cluster_label: "Entities: entity, entity, ...",
            ...
        }
    """
    cluster_tuples = {}
    for k, v in cluster_counts.items():
        cluster_tuples[k] = sorted(
            [tuple([entity, count]) for entity, count in v.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    return {
        k: "E: " + ", ".join([x[0] for x in v[:num_entities]])
        for k, v in cluster_tuples.items()
    }


# Functions to produce Journal labels
def get_entity_journal_counts(
    journal_entities: Dict[str, Sequence[str]], output: str = "absolute"
) -> Dict[str, Dict[str, int]]:
    """Creates a dictionary of entity counts at a specified level
        of the hierarchical taxonomy.

    Args:
        journal_entities (Dict[str, Sequence[str]]): A dictionary of
            journal IDs to entities.
        output (str, optional): Output type, either "absolute" or
            "relative". Defaults to "absolute".

    Returns:
        Dict[str, Dict[str, int]]: A dictionary of entity counts at a
            specified level of the hierarchical taxonomy.

        {
            entity: {
                journal: count,
                journal: count,
                ...
            },
            ...
        }
    """
    assert output in [
        "absolute",
        "relative",
    ], "output must be one of 'absolute' or 'relative'"
    journal_entity_counts = get_journal_entity_counts(journal_entities, output=output)
    entity_counts = get_entity_counts(journal_entities)

    entity_journal_counts = defaultdict(dict)
    for entity in entity_counts.keys():
        journals = [
            journal
            for journal, jentities in journal_entities.items()
            if entity in jentities
        ]
        for journal in journals:
            entity_journal_counts[entity].update(
                {journal: journal_entity_counts[journal][entity]}
            )
    return entity_journal_counts


def get_cluster_journal_counts(
    cluster_counts: Dict[str, Dict[str, int]],
    journal_entities: Dict[str, Sequence[str]],
) -> Dict[str, Dict[str, int]]:
    """Creates a dictionary of journal counts for each cluster.

    Args:
        cluster_counts (Dict[str, Dict[str, int]]): A dictionary of
            cluster labels to entity counts.
        journal_entities (Dict[str, Sequence[str]]): A dictionary of
            journal IDs to entities.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary of journal counts for
            each cluster.

        {
            cluster_label: {
                journal: count,
                journal: count,
                ...
            },
            ...
        }
    """

    entity_journal_counts = get_entity_journal_counts(journal_entities, "absolute")
    cluster_journal_prop = defaultdict(dict)
    for cluster, entities in cluster_counts.items():
        for entity, _ in entities.items():
            count = entity_journal_counts[entity]
            for journal, jcount in count.items():
                if journal in cluster_journal_prop[cluster].keys():
                    cluster_journal_prop[cluster][journal] += jcount
                else:
                    cluster_journal_prop[cluster][journal] = jcount
    return cluster_journal_prop


def get_cluster_entity_journal_counts(
    taxonomy: pd.DataFrame, journal_entities: Dict[str, Sequence[str]], level: int = 1
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Creates a dictionary of journal counts for each entity at a
        specified level of the hierarchical taxonomy.

    Args:
        taxonomy (pd.DataFrame): A dataframe of the hierarchical
            taxonomy.
        journal_entities (Dict[str, Sequence[str]]): A dictionary of
            journal IDs to entities.

    Returns:
        Dict[str, Dict[str, Dict[str, int]]]: A dictionary of journal
            counts for each entity at a specified level of the
            hierarchical taxonomy.
    """
    cluster_counts = get_level_entity_counts(journal_entities, taxonomy, level=level)
    entity_journal_counts = get_entity_journal_counts(journal_entities, "absolute")
    cluster_journal_prop = defaultdict(dict)
    for cluster, entities in cluster_counts.items():
        cluster_journal_prop[cluster] = {}
        for entity, _ in entities.items():
            cluster_journal_prop[cluster][entity] = {}
            count = entity_journal_counts[entity]
            for journal, jcount in count.items():
                cluster_journal_prop[cluster][entity][journal] = jcount
    return cluster_journal_prop


def get_cluster_journal_ranks(
    cluster_counts: Dict[str, Dict[str, int]],
    journal_entities: Dict[str, Sequence[str]],
    output: str = "relative",
) -> Dict[str, Dict[str, float]]:
    """Creates a dictionary of journal ranks for each cluster. If output
        is "absolute", the rank is the number of entities in the cluster
        that are also in the journal. If output is "relative", the rank
        is the number of entities in the cluster that are also in the
        journal, normalised by the total number of entities in the
        cluster.

    The formula for the rank is:
        rank = (
            log(2 + (cluster_entity_count / cluster_total_count))**2 *
            log(1 + (cluster_journal_count / cluster_total_count))**2 *
            (cluster_journal_entity_count / cluster_journal_total_count)
        )

    Args:
        cluster_counts (Dict[str, Dict[str, int]]): A dictionary of
            cluster labels to entity counts.
        entity_journal_counts (Dict[str, Dict[str, int]]): A dictionary
            of entity counts in journals.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary of journal ranks for
            each cluster.

            {
                cluster_label: {
                    journal: value,
                    journal: value,
                    ...
                },
                ...
            }

    """
    assert output in [
        "absolute",
        "relative",
    ], "output must be one of 'absolute' or 'relative'"
    entity_journal_prop = get_entity_journal_counts(journal_entities, "relative")
    cluster_journal_counts = get_cluster_journal_counts(
        cluster_counts,
        journal_entities,
    )

    cluster_journal_rank = defaultdict(dict)
    for clust_label, clust_dict in cluster_counts.items():
        nconst = deepcopy(sum(clust_dict.values())) if output == "relative" else 1
        if len(clust_dict.keys()) == 0:
            cluster_journal_rank[clust_label] = {}
            continue
        for entity, ecount in clust_dict.items():
            val = entity_journal_prop[entity]
            for journal, jprop in val.items():
                jcount = cluster_journal_counts[clust_label][journal]
                if journal in cluster_journal_rank[clust_label].keys():
                    cluster_journal_rank[clust_label][journal] += (
                        np.log(2 + (ecount / nconst)) ** 2
                        * np.log(1 + (jcount / nconst)) ** 2
                        * jprop
                    )
                else:
                    cluster_journal_rank[clust_label][journal] = (
                        np.log(2 + (ecount / nconst)) ** 2
                        * np.log(1 + (jcount / nconst)) ** 2
                        * jprop
                    )
    return cluster_journal_rank


def get_cluster_journal_labels(
    cluster_journal_ranks: Dict[str, Dict[str, int]], num_labels: int = 3
) -> Dict[str, str]:
    """Creates a dictionary of journal labels for each cluster.

    Args:
        num_labels (int, optional): The number of labels to return.
            Defaults to 3.

    Returns:
        Dict[str, str]: A dictionary of journal labels for each cluster.

            {
                cluster_label: "Journals: journal, journal, journal",
                ...
            }
    """
    cluster_tuples = {}
    for k, v in cluster_journal_ranks.items():
        cluster_tuples[k] = sorted(
            [tuple([entity, count]) for entity, count in v.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    return {
        k: "Journals: " + ", ".join([x[0] for x in v[:num_labels]])
        for k, v in cluster_tuples.items()
    }


def get_main_entity_journal(
    entity: str, entity_journal_counts: Dict[str, Dict[str, int]]
) -> str:
    """Returns the journal with the most entities for a given entity.

    Args:
        entity (str): The entity to get the journal for.
        entity_journal_counts (Dict[str, Dict[str, int]]): A dictionary
            of entity counts in journals.

    Returns:
        str: The journal with the most entities for a given entity.
    """
    journal_counts = entity_journal_counts[entity]
    if len(journal_counts.keys()) == 0:
        return "Other"
    max_count = max(journal_counts.values())
    max_journals = [k for k, v in journal_counts.items() if v == max_count]
    return random.choice(max_journals)


def build_labelled_taxonomy(
    df: pd.DataFrame,
    journal_entities: Dict[str, Sequence[str]],
    entity_counts: Dict[str, int],
    entity_journal_counts: Dict[str, Dict[str, int]],
    levels: int = 3,
) -> pd.DataFrame:
    """Builds a labelled taxonomy from a taxonomy dataframe, a
        dictionary of journal to entities, and a dictionary of
        cluster to entity counts.

    Args:
        df (pd.DataFrame): A taxonomy dataframe, with at least one
            Level column and one Entity column.
        journal_entities (Dict[str, Sequence[str]]): A dictionary
            of journal to entities.
        cluster_counts (Dict[str, int]): A dictionary of entity counts
            in journals.
        entity_journal_counts (Dict[str, Dict[str, int]]): A
            dictionary of entity to journal counts.+
        levels (int, optional): The number of levels to label. Defaults
            to 3.

    Returns:
        pd.DataFrame: A labelled taxonomy dataframe.
    """
    labelled_taxonomy = deepcopy(df)
    for level in list(range(1, 1 + levels)):
        logger.info(f"Level {str(level)} - Getting level cluster counts")
        cluster_counts = get_level_entity_counts(journal_entities, df, level=level)

        logger.info(f"Level {str(level)} - Getting level entity labels")
        cluster_labels = get_cluster_entity_labels(cluster_counts, num_entities=2)

        logger.info(f"Level {str(level)} - Getting level journal labels")
        journal_labels = pipe(
            get_cluster_journal_ranks(cluster_counts, journal_entities),
            get_cluster_journal_labels,
        )

        logger.info(f"Level {str(level)} - Adding labels to taxonomy")
        labelled_taxonomy[
            "Level_{}_Entity_Labels".format(str(level))
        ] = labelled_taxonomy["Level_{}".format(str(level))].map(cluster_labels)
        labelled_taxonomy[
            "Level_{}_Journal_Labels".format(str(level))
        ] = labelled_taxonomy["Level_{}".format(str(level))].map(journal_labels)

    logger.info(f"All levels complete - Adding main journal labels")
    labelled_taxonomy["Main_Journal"] = labelled_taxonomy.index.map(
        lambda x: get_main_entity_journal(
            entity=x, entity_journal_counts=entity_journal_counts
        )
    )

    logger.info(f"All levels complete - Adding entity counts")
    labelled_taxonomy["Entity_Count"] = labelled_taxonomy.index.map(
        lambda x: entity_counts[x] if entity_counts[x] != 0 else 1
    )

    logger.info(f"All levels complete - Adding journal counts")
    labelled_taxonomy["Journal_Counts"] = labelled_taxonomy.index.map(
        lambda x: entity_journal_counts[x]
    )

    logger.info(f"All levels complete - Normalising labels (empty -> other)")
    labelled_taxonomy.replace(
        "Journals: $|Entities: $", "Other", regex=True, inplace=True
    )
    return labelled_taxonomy
