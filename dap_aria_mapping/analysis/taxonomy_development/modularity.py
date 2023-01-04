from dap_aria_mapping import BUCKET_NAME
from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
    get_taxonomy_config,
)
from dap_aria_mapping.getters.cooccurrence_network import get_cooccurrence_network
from networkx.algorithms.community import modularity

if __name__ == "__main__":
    print("loading in results file")
    taxonomy = get_cooccurrence_taxonomy()
    taxonomy.reset_index(inplace=True)

    print("loading in network")
    network = get_cooccurrence_network()

    # load taxonomy config parameters
    config = get_taxonomy_config()["community_detection"]
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
