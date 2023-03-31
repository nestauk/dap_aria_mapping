from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME
import networkx as nx
import polars as pl

def get_collaboration_network(dataset: str) -> nx.Graph:
    """get the collaboration network for either patents or publications

    Args:
        dataset (str): "Academia" for publication network or "Industry" for patent network 

    Returns:
        nx.Graph: collaboration network
    """
    if dataset == "Industry":
        return download_obj(BUCKET_NAME, "outputs/app_data/change_makers/networks/patents_institutions.pkl")
    elif dataset == "Academia":
        return download_obj(BUCKET_NAME, "outputs/app_data/change_makers/networks/publications_institutions.pkl")
    else:
        return "Not implemented"

def get_topic_domain_area_lookup(doc_type: str) -> pl.DataFrame:
    """get a table of all topic (level 3), area (level 2), and domain (level 1) names (via chatgpt) to use to populate filters.
        Uses either topics present in patents, publications, or both

    Args:
        doc_type(str): "patents", "publications", or "both"

    Returns:
        pl.DataFrame: all unique topics and their corresponding area and domain names
    """
    if doc_type == "patents":
        return pl.DataFrame(download_obj(BUCKET_NAME, "outputs/app_data/patent_topic_filter_lookup.parquet", download_as = "dataframe"))
    elif doc_type == "publications":
        return pl.DataFrame(download_obj(BUCKET_NAME, "outputs/app_data/publication_topic_filter_lookup.parquet", download_as = "dataframe"))
    elif doc_type == "both":
        return pl.concat([
            pl.DataFrame(download_obj(BUCKET_NAME, "outputs/app_data/patent_topic_filter_lookup.parquet", download_as = "dataframe")),
            pl.DataFrame(download_obj(BUCKET_NAME, "outputs/app_data/publication_topic_filter_lookup.parquet", download_as = "dataframe"))
        ]).unique(subset = "topic")
    else:
        print("Invalid doctype option")

def get_quad_chart_data() -> pl.DataFrame:
    """gets the disruption and volume by institution to populate the quad chart

    Returns:
        pl.DataFrame: polars dataframe with columns ["domain_name", "area_name", "topic_name", "affiliation_string", "volume", "average_cd_score"]
    """
    return pl.DataFrame(
        download_obj(
            BUCKET_NAME, 
            "outputs/app_data/change_makers/disruption_by_institution.parquet",
            download_as = "dataframe"
            )
        )
