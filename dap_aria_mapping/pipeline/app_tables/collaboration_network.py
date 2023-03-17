import argparse
from dap_aria_mapping.getters.openalex import get_openalex_institutes
from dap_aria_mapping.getters.patents import get_patents, get_patent_topics
from dap_aria_mapping.getters.taxonomies import get_topic_names
import polars as pl
import pandas as pd
import networkx as nx
from dap_aria_mapping.utils.app_data_utils import expand_topic_col, add_area_domain_chatgpt_names
from dap_aria_mapping import BUCKET_NAME, logger
from typing import List, Dict, Tuple
from itertools import combinations, chain
from collections import Counter, defaultdict
from nesta_ds_utils.loading_saving.S3 import upload_obj

def docs_per_org(orgs_df: pl.DataFrame) -> Dict[str, int]:
    q = (
        orgs_df.lazy()
        .unique(subset = ["publication_number","assignee_harmonized_names"])
        .groupby("assignee_harmonized_names")
        .agg([pl.count("publication_number")])
        )
    return q.collect().to_pandas().set_index("assignee_harmonized_names").to_dict(orient = "index")


def node_data(orgs_df_with_topics: pl.DataFrame, by: str = "topic") -> Dict[str, List[str]]:
    q = (
    orgs_df_with_topics.lazy()
    .groupby("assignee_harmonized_names")
    .agg([pl.col(by)])
    )
    df = q.collect()
    return df.to_pandas().set_index("assignee_harmonized_names").to_dict(orient = "index")

def get_edges(orgs_df: pl.DataFrame) -> Dict[Tuple[str,str], int]:
    q = (
        orgs_df.lazy()
        .unique(subset = ["publication_number","assignee_harmonized_names"])
        .groupby("publication_number")
        .agg([pl.col("assignee_harmonized_names")])
        )
    df = q.collect()
    sequences = df["assignee_harmonized_names"].to_list()
    combos = list(combinations(sorted(set(sequence)), 2) for sequence in sequences)
    return Counter(chain(*combos))

def edge_data(orgs_df_with_topics: pl.DataFrame, by: str = "topic") -> Dict[str, Dict[str, int]]:
    q = (
        orgs_df_with_topics.lazy()
        .groupby([by, "publication_number"])
        .agg([
            pl.col("assignee_harmonized_names")
            ])
        )
    df = q.collect()
    unique_topics = df[by].unique().to_list()
    edge_data = defaultdict(lambda: defaultdict(int))
    for topic in unique_topics:
        topic_data = df.filter(pl.col(by) == topic)
        sequences = topic_data["assignee_harmonized_names"].to_list()
        combos = list(combinations(sorted(set(sequence)), 2) for sequence in sequences)
        topic_combos = Counter(chain(*combos))
        
        for combo in topic_combos:
            edge_data[combo][topic] +=1
    return edge_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--doc_type",
        default="patents",
        type=str,
        help="options: patents or publications",
    )

    parser.add_argument(
        "--level",
        default = "institutions"
        type=str,
        help = "options: institutions or individuals"
    )

    args = parser.parse_args()

    #if args.doc_type == "publications":
    #    docs = get_openalex_institutes()
    
    #elif args.doc_type == "patents" and args.level == "institution":
    logger.info("Loading Patent Data")
    docs = pl.DataFrame(get_patents())[["publication_number", "assignee_harmonized_names", "assignee_harmonized_names_types"]]
    patents_with_topics = get_patent_topics(tax = "cooccur", level = 3)
    patents_with_topics_df = pl.DataFrame(
        pd.DataFrame.from_dict(patents_with_topics, orient='index'
        ).T.unstack().dropna().reset_index(drop=True, level=1).to_frame().reset_index())
    patents_with_topics_df.columns = ["publication_number", "topic"]
    
    
    logger.info("Formatting patents data")
    #add assignees and the assignee types to patents tagged with topics
    df = patents_with_topics_df.join(docs, how = "left", on = "publication_number")

    #explode assignee name and type columns so each row contains a single assignee and the associated type vs a list
    #NOTE: I DONT THINK THIS WORKS PROPERLY AS THE LISTS OF NAME TYPES SEEM TO BE OUT OF ORDER OF THE NAMES (see patent WO-2012141480-A2)
    df = df.filter(
        pl.col("assignee_harmonized_names").arr.lengths() == pl.col("assignee_harmonized_names_types").arr.lengths()
        ).explode(
            ["assignee_harmonized_names", "assignee_harmonized_names_types"]
            )
    #only consider organisations (not individuals)
    orgs_df = df.filter(pl.col("assignee_harmonized_names_types") == "ORGANISATION")

    #add area and domain col
    orgs_df_with_topics = add_area_domain_chatgpt_names(
        expand_topic_col(orgs_df).unique(subset=["publication_number", "topic", "assignee_harmonized_names"]))

    logger.info("GENERATING NETWORK")
    network = nx.Graph()

    logger.info("Adding nodes with the following attributes: topics, areas, domains")
    network.add_nodes_from(set(orgs_df["assignee_harmonized_names"]))
    nx.set_node_attributes(network, docs_per_org(orgs_df), "overall_doc_count")
    nx.set_node_attributes(network, node_data(orgs_df_with_topics, "topic_name"), "topics")
    nx.set_node_attributes(network, node_data(orgs_df_with_topics, "area_name"), "areas")
    nx.set_node_attributes(network, node_data(orgs_df_with_topics, "domain_name"), "domains")

    logger.info("Adding edges")
    edges = get_edges(orgs_df)
    network.add_edges_from(list(edges.keys()))
    nx.set_edge_attributes(network, edges, "overall_doc_count")
    nx.set_edge_attributes(network, edge_data(orgs_df_with_topics, by="topic_name"), "topics")
    nx.set_edge_attributes(network, edge_data(orgs_df_with_topics, by="area_name"), "areas")
    nx.set_edge_attributes(network, edge_data(orgs_df_with_topics, by="domain_name"), "domains")

    logger.info("Saving network to S3")
    upload_obj(network, BUCKET_NAME, "outputs/app_data/change_makers/networks/{}_{}.pkl".format(args.doc_type, args.level))
    


