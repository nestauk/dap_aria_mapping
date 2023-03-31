import argparse
from dap_aria_mapping.getters.openalex import get_openalex_institutes, get_openalex_topics, get_openalex_authorships
from dap_aria_mapping.getters.patents import get_patents, get_patent_topics
import polars as pl
import pandas as pd
import networkx as nx
import pickle
from dap_aria_mapping.utils.app_data_utils import expand_topic_col, add_area_domain_chatgpt_names
from dap_aria_mapping import BUCKET_NAME, logger
from typing import List, Dict, Tuple, Union
from itertools import combinations, chain
from collections import Counter, defaultdict
from nesta_ds_utils.loading_saving.S3 import upload_obj

def docs_per_org(
    orgs_df: pl.DataFrame, 
    id_col: str = "publication_number", 
    org_name_col: str = "assignee_harmonized_names",
    top_n: Union[int, bool] = False) -> Union[Dict[str, int], Tuple[Dict[str,int], List[str]]]:
    """Creates a dataframe with the total number of documents generated per institution/organisation. 
        Used to generate node weights.

    Args:
        orgs_df (pl.DataFrame): dataframe with institution/organisation names and document ids
        id_col (str, optional): column to indicate a document id. Defaults to "publication_number".
        org_name_col (str, optional): column to indicate the institution/organisation names. Defaults to "assignee_harmonized_names".
        top_n (Union[int, bool], optional): if this is included, filter the results to only include the top_n organisations by count of total documents

    Returns:
        Dict[str, int]: key: institution/organisastion name, value: total number of documents produced
    """
    q = (
        orgs_df.lazy()
        .unique(subset = [id_col,org_name_col])
        .groupby(org_name_col)
        .agg([pl.count(id_col)])
        .sort(id_col, descending = True, nulls_last = True)
        )
    if top_n:
        df = q.collect().limit(top_n).to_pandas()
        top_orgs = df[org_name_col].to_list()
        return (df.set_index(org_name_col).to_dict(orient = "index"), top_orgs)
    else:
        return q.collect().to_pandas().set_index(org_name_col).to_dict(orient = "index")


def node_data(
    orgs_df_with_topics: pl.DataFrame, 
    by: str = "topic", 
    org_name_col: str = "assignee_harmonized_names") -> Dict[str, List[str]]:
    """generates a dictionary of domains, areas, or topics (specified with by) that an organisation/institution has
    produced at least one document in. Used to populate node attributes.

    Args:
        orgs_df_with_topics (): dataframe with topic, area, and domain names and document ids
        by (str, optional): either "domain", "area", or "topic". Specified what level to aggregate by. Defaults to "topic".
        org_name_col (str, optional): column that specified organisation/institution name. Defaults to "assignee_harmonized_names".

    Returns:
        Dict[str, List[str]]: 
            key: organisation/institution name, 
            value: list of either domains, areas, or topics associated with the org/institution
    """
    q = (
    orgs_df_with_topics.lazy()
    .groupby(org_name_col)
    .agg([pl.col(by)])
    )
    df = q.collect()
    return df.to_pandas().set_index(org_name_col).to_dict(orient = "index")

def get_edges(
    orgs_df: pl.DataFrame,
    id_col: str = "publication_number", 
    org_name_col: str = "assignee_harmonized_names") -> Dict[Tuple[str,str], int]:
    """gets all pairs of orgs/institutions that have collaborated at least once on a document.
        Used to created edges.

    Args:
        orgs_df (pl.DataFrame): dataframe with institution/organisation names and document ids
        id_col (str, optional): column to indicate a document id. Defaults to "publication_number".
        org_name_col (str, optional): column to indicate organisation/institution name. Defaults to "assignee_harmonized_names".

    Returns:
        Dict[Tuple[str,str], int]: key: pairs of institutions/organisations, value: total documents they have collaborated on
    """
    q = (
        orgs_df.lazy()
        .unique(subset = [id_col, org_name_col])
        .groupby(id_col)
        .agg([pl.col(org_name_col)])
        )
    df = q.collect()
    sequences = df[org_name_col].to_list()
    combos = list(combinations(sorted(set(sequence)), 2) for sequence in sequences)
    return Counter(chain(*combos))

def edge_data(
    orgs_df_with_topics: pl.DataFrame,
    by: str = "topic",
    id_col: str = "publication_number", 
    org_name_col: str = "assignee_harmonized_names") -> Dict[str, Dict[str, int]]:
    """calculates the total number of documents within a domain, area, topic that a pair of institutions/orgs have collaborated on
        Used to add metadata to the edges for filtering.

    Args:
        orgs_df_with_topics (pl.DataFrame): dataframe with topic, area, and domain names and document ids
        by (str, optional): either "domain", "area", or "topic". Specified what level to aggregate by. Defaults to "topic".
        id_col (str, optional): column to indicate a document id.. Defaults to "publication_number".
        org_name_col (str, optional): column to indicate organisation/institution name. Defaults to "assignee_harmonized_names".

    Returns:
        Dict[Tuple[str, str], Dict[str, int]]: 
            key: pair of institutions/organisations
            value: 
                key: area/domain/topic name as specified with "by"
                value: count of number of times the pair has collaborated on the topic/area/domain
    """
    q = (
        orgs_df_with_topics.lazy()
        .groupby([by, id_col])
        .agg([
            pl.col(org_name_col)
            ])
        )
    df = q.collect()
    unique_topics = df[by].unique().to_list()
    edge_data = defaultdict(lambda: defaultdict(int))
    for topic in unique_topics:
        topic_data = df.filter(pl.col(by) == topic)
        sequences = topic_data[org_name_col].to_list()
        combos = list(combinations(sorted(set(sequence)), 2) for sequence in sequences)
        topic_combos = Counter(chain(*combos))
        
        for combo in topic_combos:
            edge_data[combo][topic] +=1
    return edge_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--doc_type",
        default="patents",
        type=str,
        help="options: patents or publications",
    )

    parser.add_argument(
        "--level",
        default = "institutions",
        type=str,
        help = "options: institutions or individuals"
    )

    parser.add_argument(
        "--top_n",
        default= False,
        type = int,
        required=False
    )

    args = parser.parse_args()

    if args.doc_type == "publications" and args.level == "institutions":
        logger.info("Loading openalex institutes data")
        authorships = pl.DataFrame(
            get_openalex_authorships()).select([pl.col("id"), pl.col("affiliation_string")]
        )
        id_col = "id"
        org_name_col = "affiliation_string"

        logger.info("Loading publications with topics")
        pubs_with_topics_df = pl.DataFrame(
            pd.DataFrame.from_dict(
                get_openalex_topics(tax="cooccur", level=3), orient='index')
                .T
                .unstack()
                .dropna()
                .reset_index(drop=True, level=1)
                .to_frame()
                .reset_index()
                )
        pubs_with_topics_df.columns = [id_col, "topic"]

        # add institutions affiliated with authors to publications tagged with topics
        # filter out publications that we don't have org names for
        # NOTE: WE HAVE ORG NAMES FOR 2,491,974 OUT OF 2,522,837 PUBLICATIONS
        orgs_df = pubs_with_topics_df.join(
            authorships, on=id_col, how="left"
        ).filter(
            ~pl.all(pl.col(org_name_col).is_null())
        )
    
    elif args.doc_type == "patents" and args.level == "institutions":
        id_col = "publication_number"
        org_name_col = "assignee_harmonized_names"
        logger.info("Loading Patent Data")
        docs = pl.DataFrame(get_patents())[
            [id_col, org_name_col, "assignee_harmonized_names_types"]
        ]
        patents_with_topics = get_patent_topics(tax="cooccur", level=3)
        patents_with_topics_df = pl.DataFrame(
            pd.DataFrame.from_dict(
                patents_with_topics, orient='index'
            )
            .T
            .unstack()
            .dropna()
            .reset_index(drop=True, level=1)
            .to_frame()
            .reset_index()
        )

        patents_with_topics_df.columns = [id_col, "topic"]
        
        logger.info("Formatting patents data")
        #add assignees and the assignee types to patents tagged with topics
        df = patents_with_topics_df.join(docs, how="left", on=id_col)

        #explode assignee name and type columns so each row contains a single assignee and the associated type vs a list
        #NOTE: I DONT THINK THIS WORKS PROPERLY AS THE LISTS OF NAME TYPES SEEM TO BE OUT OF ORDER OF THE NAMES (see patent WO-2012141480-A2)
        df = df.filter(
            pl.col(org_name_col).arr.lengths() == pl.col("assignee_harmonized_names_types").arr.lengths()
        ).explode(
            [org_name_col, "assignee_harmonized_names_types"]
        )
        #only consider organisations (not individuals)
        orgs_df = df.filter(pl.col("assignee_harmonized_names_types") == "ORGANISATION")

    #add area and domain col
    orgs_df_with_topics = add_area_domain_chatgpt_names(
        expand_topic_col(orgs_df).unique(subset=[id_col, "topic", org_name_col])
    )

    logger.info("GENERATING NETWORK")
    network = nx.Graph()

    logger.info("Adding nodes with the following attributes: topics, areas, domains")
    if args.top_n:
        print("Only including top {} orgs".format(args.top_n))
        node_metadata, nodes =  docs_per_org(orgs_df, id_col, org_name_col, top_n=args.top_n)
        orgs_df = orgs_df.filter(pl.col(org_name_col).is_in(nodes))
        orgs_df_with_topics = orgs_df_with_topics.filter(pl.col(org_name_col).is_in(nodes))
        network.add_nodes_from(nodes)
    else:
        node_metadata = docs_per_org(orgs_df, id_col, org_name_col)
        network.add_nodes_from(set(orgs_df[org_name_col]))

    nx.set_node_attributes(network,node_metadata, "overall_doc_count")
    nx.set_node_attributes(network, node_data(orgs_df_with_topics, "topic_name", org_name_col), "topics")
    nx.set_node_attributes(network, node_data(orgs_df_with_topics, "area_name", org_name_col), "areas")
    nx.set_node_attributes(network, node_data(orgs_df_with_topics, "domain_name", org_name_col), "domains")

    logger.info("Adding edges")
    edges = get_edges(orgs_df, id_col, org_name_col)
    network.add_edges_from(list(edges.keys()))
    nx.set_edge_attributes(network, edges, "overall_doc_count")
    nx.set_edge_attributes(network, edge_data(orgs_df_with_topics, by="topic_name", id_col= id_col, org_name_col = org_name_col), "topics")
    nx.set_edge_attributes(network, edge_data(orgs_df_with_topics, by="area_name", id_col= id_col, org_name_col= org_name_col), "areas")
    nx.set_edge_attributes(network, edge_data(orgs_df_with_topics, by="domain_name", id_col= id_col, org_name_col= org_name_col), "domains")

    print("Saving network")
    upload_obj(network, BUCKET_NAME, "outputs/app_data/change_makers/networks/{}_{}.pkl".format(args.doc_type, args.level))



