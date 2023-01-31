from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME, PROJECT_DIR
from typing import Dict
import yaml


def get_tree_depths(tax_name: str) -> Dict[str, Dict[str, int]]:
    """gets a dictionary with metrics about the tree depth of the given taxonomies

    Args:
        tax_name (str): name of taxonomy to load results. Options are: cooccur, centroids, imbalanced

    Args:
        tax_name (str): name of taxonomy to load results. Options are: cooccur, centroids, imbalanced 

    Returns:
        dict: size and depth metrics of trees in taxonomy
    """
    return download_obj(
        BUCKET_NAME,
        "outputs/validation_metrics/taxonomy_depth/{}.json".format(tax_name),
        download_as="dict",
    )


def get_entropy(tax_name: str) -> Dict[str, int]:
    """gets a dictionary with metrics about the entropy of the distribution of
        entites/category at a given level of the taxonomy

    Args:
        tax_name (str): name of taxonomy to load results. Options are: cooccur, centroids, imbalanced

    Returns:
        dict: key: taxonomy name, value: dict: key: level, value: entropy of frequency distribution
    """
    return download_obj(
        BUCKET_NAME,
        "outputs/validation_metrics/entropy/{}.json".format(tax_name),
        download_as="dict",
    )


def get_chisq(tax_name: str) -> Dict[str, int]:
    """gets a dictionary with the chi square test statistic that compares the frequency
        distribution of entities/category to a uniform distribution

    Args:
        tax_name (str): name of taxonomy to load results. Options are: cooccur, centroids, imbalanced

    Returns:
        dict: key: taxonomy name, value: dict: key: level, value: chisq stat
    """
    return download_obj(
        BUCKET_NAME,
        "outputs/validation_metrics/chisq/{}.json".format(tax_name),
        download_as="dict",
    )


def get_pairwise_depth(tax_name: str, level: str) -> Dict[str, int]:
    """gets a dictionary with the chi square test statistic that compares the frequency
        distribution of entities/category to a uniform distribution

    Args:
        tax_name (str): name of taxonomy to load results. Options are: cooccur, centroids, imbalanced
        level (str): either 'topic' or 'subtopic'

    Returns:
        dict: key: taxonomy name, value: dict: key: level, value: chisq stat
    """
    return download_obj(
        BUCKET_NAME,
        "outputs/validation_metrics/pairwise_depths/{}_level/{}.json".format(
            level, tax_name
        ),
        download_as="dict",
    )


def get_topic_groups() -> Dict[str, Dict[str, list]]:
    """gets groupings of known entities within topics from
    https://en.wikipedia.org/wiki/Outline_of_academic_disciplines#
    Returns:
        Dict: key: high level topic, value: key: discipline, value: list of entities
    """
    with open(
        PROJECT_DIR / "dap_aria_mapping" / "config" / "entity_groups_topics.yaml", "r"
    ) as f:
        config = yaml.safe_load(f)
    return config


def get_subtopic_groups() -> Dict[str, Dict[str, Dict[str, list]]]:
    """gets groupings of known entities grouped by the DAP team
    Returns:
        Dict: key: high level topic, value: key: discipline, value: key: topic, value: list of subtopics
    """
    with open(
        PROJECT_DIR / "dap_aria_mapping" / "config" / "entity_groups_subtopics.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
    return config
