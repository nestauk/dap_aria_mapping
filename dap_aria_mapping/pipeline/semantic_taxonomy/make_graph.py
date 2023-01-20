import boto3, pickle, argparse
import pandas as pd
import numpy as np
import networkx as nx
from toolz import pipe
from dap_aria_mapping import BUCKET_NAME, logger
from dap_aria_mapping.getters.taxonomies import get_cooccurrences


def process_dataframe(df: pd.DataFrame) -> nx.Graph:
    """Transforms a pandas dataframe into a graph adjacency matrix

    Args:
        df (pd.DataFrame): pandas dataframe to transform

    Returns:
        nx.Graph: graph object
    """
    logger.info("Creating dictionary of network tags")
    network_dict = dict(zip(df.index, range(len(df.index))))

    logger.info("Normalising dataframe")
    df = pipe(
        df,
        lambda df: (
            df.astype(np.int8).rename(columns=network_dict, index=network_dict)
        ),
    )

    logger.info("Creating network object")
    network = nx.Graph()
    network.add_nodes_from(meta_cluster_df.index)

    logger.info("Creating edge dictionaries and update network")

    for idx, (tag, row) in enumerate(df.iterrows()):
        triu_row = row.iloc[(1 + idx) :]
        ndict = (triu_row[triu_row > 2]).to_dict()
        for nkey, nval in ndict.items():
            network.add_edge(tag, nkey, weight=nval)
    return network, network_dict


s3 = boto3.client("s3")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket_name",
        type=str,
        default=BUCKET_NAME,
        help="Name of S3 bucket to save outputs to",
    )
    parser.add_argument(
        "--cluster_object",
        type=str,
        default="meta_cluster",
        help="Name of S3 object to read meta cluster data from",
    )
    args = parser.parse_args()

    logger.info("Reading meta cluster object into pandas dataframe")
    meta_cluster_df = get_cooccurrences(
        cluster_object=args.cluster_object, bucket_name=args.bucket_name
    )

    logger.info("Transforming dataframe into graph object")
    network, network_dict = process_dataframe(meta_cluster_df)

    logger.info("Saving graph object to S3")
    s3.put_object(
        Body=pickle.dumps(network),
        Bucket=args.bucket_name,
        Key=f"outputs/semantic_taxonomy/{args.cluster_object}_network.pkl",
    )
    s3.put_object(
        Body=pickle.dumps(network),
        Bucket=args.bucket_name,
        Key=f"outputs/semantic_taxonomy/{args.cluster_object}_network_dict.pkl",
    )
