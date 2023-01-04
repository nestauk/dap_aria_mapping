from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME
from networkx.algorithms.community import modularity
import yaml

if __name__ == "__main__":
    print("loading in results file")
    taxonomy = download_obj(
        BUCKET_NAME,
        "outputs/community_detection_clusters.parquet",
        download_as="dataframe",
    )
    taxonomy.reset_index(inplace=True)

    print("loading in network")
    network = download_obj(BUCKET_NAME, "outputs/cooccurrence_network.pkl")

    # load taxonomy config file with parameters
    with open("dap_aria_mapping/config/taxonomy.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)["community_detection"]
    resolution = config["starting_resolution"]
    resolution_increments = config["resolution_increments"]

    total_mod = 0
    counter = 0

    for i in range(1, len(taxonomy.columns)):
        communities = taxonomy.groupby(taxonomy.columns[i])["Entity"].apply(list)
        score = modularity(
            network,
            communities,
            weight="association_strength",
            resolution=resolution,
        )
        resolution += resolution_increments
        total_mod += score
        counter += 1
        print("Modularity at level {}: {}".format(i, score))

    print(
        "Average Modularity of {} Layer Taxonomy: {}".format(
            counter, (total_mod / counter)
        )
    )
