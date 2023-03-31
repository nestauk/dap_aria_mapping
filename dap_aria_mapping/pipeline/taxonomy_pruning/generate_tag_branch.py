import pickle
from dap_aria_mapping import PROJECT_DIR
from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
)
from dap_aria_mapping.utils.topic_names import *
from dap_aria_mapping import manual_checks

OUTPUT_DIR_PREPRUNE = PROJECT_DIR / "outputs" / "interim" / "preprune_topics"
OUTPUT_DIR_PREPRUNE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_TAG = PROJECT_DIR / "outputs" / "interim" / "country_tags"
OUTPUT_DIR_TAG.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    entities = get_cooccurrence_taxonomy().index.tolist()
    with open(OUTPUT_DIR_PREPRUNE / "prepruned_entities.pkl", "rb") as f:
        prepruned_entities = pickle.load(f)

    (
        entities_to_keep,
        entity_reassign,
        country_remove,
        country_reassign,
        continent_country,
    ) = [
        manual_checks[x]
        for x in [
            "ENTITY_KEEP",
            "ENTITY_REASSIGN",
            "COUNTRY_REMOVE",
            "COUNTRY_REASSIGN",
            "CONTINENT_TO_COUNTRY",
        ]
    ]
    discarded_entities = [x for x in prepruned_entities if x not in entities_to_keep]
    preprocessed_entities = [x for x in entities if x not in discarded_entities]

    with open(OUTPUT_DIR_PREPRUNE / "discarded_entities.pkl", "wb") as f:
        pickle.dump(discarded_entities, f)
    with open(OUTPUT_DIR_PREPRUNE / "preprocessed_entities.pkl", "wb") as f:
        pickle.dump(preprocessed_entities, f)
    with open(OUTPUT_DIR_TAG / "tagged_entities.pkl", "rb") as f:
        tagged_entities = pickle.load(f)

    tag_df = pd.DataFrame(tagged_entities, columns=["country", "entities"])
    tag_df = (
        tag_df.groupby("country")["entities"]
        .apply(lambda x: [item for sublist in x for item in sublist])
        .reset_index()
    )

    tag_df = tag_df.explode("entities", ignore_index=True)

    # reassign entities to countries
    tag_df["entities"] = tag_df["entities"].replace(entity_reassign)

    # remove countries
    tag_df = tag_df[~tag_df["country"].isin(country_remove)]

    # relabel countries
    tag_df["country"] = tag_df["country"].replace(country_reassign)

    # assign continent to countries
    tag_df["continent"] = tag_df["country"].apply(
        lambda x: [k for k, v in continent_country.items() if x in v][0]
    )

    with open(OUTPUT_DIR_TAG / "clean_tagged_entities.pkl", "wb") as f:
        pickle.dump(tag_df, f)
