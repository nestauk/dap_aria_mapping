"""
This file is used to manually prune the taxonomy. It is used to:
    - remove entities from the taxonomy
    - remove topics from the taxonomy
    - remove areas from the taxonomy
    - merge topics
    - move entities between topics
    - split topics
    - adjust areas
    - move areas
    - adjust topic names
    - add country tags
    - add continent tags

The manual checks are stored in a YAML file in the manual_checks folder. The YAML file is loaded
into a dictionary, and the dictionary is used to make the changes to the taxonomy.

The pruning begins at the second level of the taxonomy, as the first level is the root node.

Note that the manual checks will cause some topic names to break (these are unreliable when pruned),
so a second round of chatgpt querying is necessary following the manual pruning.

"""

from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
    get_semantic_taxonomy,
    get_topic_names,
)
from nesta_ds_utils.loading_saving.S3 import upload_obj
from dap_aria_mapping import logger, PROJECT_DIR, manual_checks, BUCKET_NAME
import pandas as pd

pd.options.display.max_colwidth = 1000
import numpy as np
import pickle, argparse
from itertools import chain


def output_duplicates(dataframe) -> None:
    """Iteratively yields duplicate rows in a dataframe. These are ought to
    be manually checked and removed.

    Args:
        dataframe (pd.DataFrame): A dataframe containing duplicate rows in the
            name outputs from chatGPT.

    Yields:
        (pd.DataFrame, str): A dataframe containing the duplicate rows and a prompt
        for checking names with BingAI.
    """
    duplis = names_df.loc[names_df.duplicated(["name"], keep=False)].sort_values(
        by=["name"]
    )
    duplicates = duplis.name.unique()
    for duplicate in duplicates:
        items = dataframe.loc[dataframe["name"] == duplicate]
        topics = items["entity_names"].to_list()
        prompt = (
            "Can you provide different topic names to these lists of related entities: \n\n"
            + "\n\n".join([f"List: {i+1}: {topics[i]}" for i in range(len(items))])
        )
        yield items, prompt


OUTPUT_DIR_PREPRUNE = PROJECT_DIR / "outputs" / "interim" / "preprune_topics"
OUTPUT_DIR_TAG = PROJECT_DIR / "outputs" / "interim" / "country_tags"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    tax_df = get_cooccurrence_taxonomy(postproc=False)

    # Domain adjustments
    logger.info("Adjusting domains")
    LEVEL_CHOICE = 2

    entity_names = get_topic_names(
        taxonomy_class="cooccur",
        name_type="entity",
        level=LEVEL_CHOICE,
        n_top=35,
    )

    new_topic_names = get_topic_names(
        taxonomy_class="cooccur",
        name_type="chatgpt",
        level=LEVEL_CHOICE,
        n_top=35,
    )

    # load country tagged entities and preprocessed entities
    with open(OUTPUT_DIR_TAG / "clean_tagged_entities.pkl", "rb") as f:
        clean_tagged_entities = pickle.load(f)
    with open(OUTPUT_DIR_PREPRUNE / "preprocessed_entities.pkl", "rb") as f:
        preprocessed_entities = pickle.load(f)

    entity_names = {
        k: v for k, v in entity_names.items() if k in new_topic_names.keys()
    }
    names_df = pd.DataFrame.from_dict(entity_names, orient="index").rename(
        columns={0: "entity_names"}
    )
    names_df = names_df.merge(
        pd.DataFrame.from_dict(new_topic_names, orient="index"),
        left_index=True,
        right_index=True,
    )

    (  # load relevant YAML changes
        adjustments,
        discard_entities,
        discard_topics,
        discard_areas,
        merge_topics,
        move_entities,
        split_topics,
        adjust_areas,
        move_areas,
    ) = [
        manual_checks[x]
        for x in [
            "ADJUSTMENTS",
            "DISCARD_ENTITIES",
            "DISCARD_TOPICS",
            "DISCARD_AREAS",
            "MERGE_TOPICS",
            "MOVE_ENTITIES",
            "SPLIT_TOPICS",
            "ADJUST_AREAS",
            "MOVE_AREAS",
        ]
    ]

    # remove all chatgpt flagged discards
    auto_discard_entities = [x for x in names_df["discard"] if x is not None]
    auto_discard_entities = [
        x for x in auto_discard_entities if "International" not in x
    ]
    discard_entities = list(set(list(chain(*auto_discard_entities, discard_entities))))

    # rename topics using adjustments and fill na with old value
    names_df["name"] = names_df.index.map(
        lambda x: adjustments.get(x, names_df.loc[x, "name"])
    )

    # remove entities from tax
    tax_df = tax_df[
        (~tax_df.index.isin(discard_entities))
        & (tax_df.index.isin(preprocessed_entities))
    ]

    # remove entities with "(film)" in them
    tax_df = tax_df[~tax_df.index.str.contains("\(film\)")]

    # include and reassign country-related entities
    clean_tagged_entities = clean_tagged_entities.rename(
        columns={"entities": "Entity"}
    ).set_index("Entity")
    clean_tagged_entities["Level_1"] = "25"

    # assign continent tags as Level_2
    continent_tags = [
        "Africa",
        "Asia and Middle East",
        "Europe",
        "North America, Central America, and Caribbean",
        "South America",
        "Oceania",
        "Antarctica",
    ]
    choices = ["25_0", "25_1", "25_2", "25_3", "25_4", "25_5", "25_6"]
    conditions = [clean_tagged_entities["continent"] == tag for tag in continent_tags]
    clean_tagged_entities["Level_2"] = np.select(conditions, choices, default="25_7")

    # assign names in Level_2
    for topic_name, topic_id in list(zip(continent_tags, choices)):
        names_df.loc[topic_id] = (
            ", ".join(
                clean_tagged_entities.loc[
                    clean_tagged_entities["continent"] == topic_name
                ]["country"].unique()
            ),
            topic_name,
            100,
            None,
        )

    # assign country tags as Level_3, carry over to levels 4 and 5
    clean_tagged_entities["Level_3"] = (
        clean_tagged_entities.groupby(["Level_2"])["country"]
        .transform(lambda x: pd.factorize(x)[0])
        .astype(str)
    )
    clean_tagged_entities["Level_3"] = (
        clean_tagged_entities["Level_2"] + "_" + clean_tagged_entities["Level_3"]
    )
    clean_tagged_entities["Level_4"] = clean_tagged_entities["Level_3"]
    clean_tagged_entities["Level_5"] = clean_tagged_entities["Level_3"]
    clean_tagged_entities = clean_tagged_entities[
        ~clean_tagged_entities.index.duplicated(keep="first")
    ]
    # join to taxonomy
    tax_df = pd.concat(
        [
            tax_df,
            clean_tagged_entities[
                ["Level_1", "Level_2", "Level_3", "Level_4", "Level_5"]
            ],
        ]
    )

    # move entities using the dictionary move_entities
    for entity, new_topic in move_entities.items():
        tax_df.loc[tax_df.index == entity, "Level_1"] = new_topic[0]
        tax_df.loc[tax_df.index == entity, "Level_2"] = new_topic[1]
        tax_df.loc[tax_df.index == entity, "Level_3"] = new_topic[2]
        tax_df.loc[tax_df.index == entity, "Level_4"] = new_topic[2]
        tax_df.loc[tax_df.index == entity, "Level_5"] = new_topic[2]

    # merge topics using the list of tuples
    for topic, new_topic in merge_topics:
        if len(new_topic.split("_")) == 1:
            unique_ids = (
                tax_df.loc[tax_df["Level_1"] == new_topic]["Level_2"]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            next_id_num = (
                int(unique_ids.str.split("_").apply(lambda x: x[-1]).astype(int).max())
                + 1
            )
            level_2 = new_topic + "_" + str(next_id_num)
            tax_df.loc[
                tax_df["Level_1"] == topic,
                ["Level_1", "Level_2", "Level_3", "Level_4", "Level_5"],
            ] = (new_topic, level_2, level_2 + "_0", level_2 + "_0", level_2 + "_0")
        elif len(new_topic.split("_")) == 2:
            unique_ids = (
                tax_df.loc[tax_df["Level_2"] == new_topic]["Level_3"]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            next_id_num = (
                int(unique_ids.str.split("_").apply(lambda x: x[-1]).astype(int).max())
                + 1
            )
            level_3 = new_topic + "_" + str(next_id_num)
            tax_df.loc[
                tax_df["Level_2"] == topic,
                ["Level_1", "Level_2", "Level_3", "Level_4", "Level_5"],
            ] = (new_topic.split("_")[0], new_topic, level_3, level_3, level_3)

    # discard topics and the associated entities
    for topic in discard_topics:
        tax_df = tax_df[tax_df["Level_2"] != topic]

    for area in discard_areas:
        tax_df = tax_df[tax_df["Level_1"] != area]

    # split topics using the dictionary split_topics
    for topic, new_topics in split_topics.items():
        unique_ids = (
            tax_df.loc[tax_df["Level_1"] == topic.split("_")[0]]["Level_2"]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        next_id_num = (
            int(unique_ids.str.split("_").apply(lambda x: x[-1]).astype(int).max()) + 1
        )
        for entities, name in new_topics:
            level_2 = topic.split("_")[0] + "_" + str(next_id_num)
            tax_df.loc[
                tax_df.index.isin(entities),
                ["Level_2", "Level_3", "Level_4", "Level_5"],
            ] = (level_2, level_2 + "_0", level_2 + "_0", level_2 + "_0")
            next_id_num += 1

            # rename topic
            names_df.loc[level_2] = (", ".join(entities), name, 100, None)

    # Area adjustments
    logger.info("Adjusting areas")
    LEVEL_CHOICE = 1

    # move areas by adding "26_" at each Level
    level_2_old_areas = {"26_" + key: value for key, value in adjust_areas.items()}
    for area in move_areas:
        for old_level, new_level in list(
            zip(
                ["Level_4", "Level_3", "Level_2", "Level_1"],
                ["Level_5", "Level_4", "Level_3", "Level_2"],
            )
        ):
            tax_df.loc[tax_df["Level_1"] == area, new_level] = tax_df.loc[
                tax_df["Level_1"] == area, old_level
            ]
            tax_df.loc[tax_df["Level_1"] == area, new_level] = (
                "26_" + tax_df.loc[tax_df["Level_1"] == area, new_level]
            )
        tax_df.loc[tax_df["Level_1"] == area, "Level_1"] = "26"

    entity_names = get_topic_names(
        taxonomy_class="cooccur",
        name_type="entity",
        level=LEVEL_CHOICE,
        n_top=35,
    )

    new_topic_names = get_topic_names(
        taxonomy_class="cooccur",
        name_type="chatgpt",
        level=LEVEL_CHOICE,
        n_top=35,
    )

    names_df1 = pd.DataFrame.from_dict(entity_names, orient="index").rename(
        columns={0: "entity_names"}
    )
    names_df1 = names_df1.merge(
        pd.DataFrame.from_dict(new_topic_names, orient="index"),
        left_index=True,
        right_index=True,
    )

    names_df1["name"] = names_df1.index.map(
        lambda x: adjust_areas.get(x, names_df1.loc[x, "name"])
    )
    names_df1.loc["25"] = (
        ", ".join([str(x) for x in tax_df.loc[tax_df["Level_1"] == "25"].index]),
        "Countries",
        100,
        None,
    )
    names_df1.loc["26"] = (
        ", ".join(tax_df.loc[tax_df["Level_1"] == "26"].index),
        "Miscellaneous",
        100,
        None,
    )

    # Topic changes
    logger.info("Adjusting topics")
    LEVEL_CHOICE = 3

    entity_names = get_topic_names(
        taxonomy_class="cooccur",
        name_type="entity",
        level=LEVEL_CHOICE,
        n_top=35,
    )

    new_topic_names = get_topic_names(
        taxonomy_class="cooccur",
        name_type="chatgpt",
        level=LEVEL_CHOICE,
        n_top=35,
    )

    names_df3 = pd.DataFrame.from_dict(entity_names, orient="index").rename(
        columns={0: "entity_names"}
    )
    names_df3 = names_df3.merge(
        pd.DataFrame.from_dict(new_topic_names, orient="index"),
        left_index=True,
        right_index=True,
    )

    for level in clean_tagged_entities.Level_3.unique():
        names_df3.loc[level] = (
            None,
            clean_tagged_entities.loc[
                clean_tagged_entities.Level_3 == level, "country"
            ].values[0],
            100,
            None,
        )

    if args.save:
        tax_df = tax_df[~tax_df.index.duplicated(keep="last")]
        upload_obj(
            tax_df,
            bucket=BUCKET_NAME,
            path_to="outputs/community_detection_taxonomy/postproc_tax.parquet",
            kwargs_writing={"index": True},
        )

        names_df = names_df.loc[names_df.index.isin(tax_df["Level_2"].unique())]
        names_df = names_df[["name", "confidence", "discard"]].to_dict(orient="index")

        upload_obj(
            names_df,
            bucket=BUCKET_NAME,
            path_to=f"outputs/topic_names/postproc/level_2.json",
        )

        names_df1 = names_df1.loc[names_df1.index.isin(tax_df["Level_1"].unique())]
        names_df1 = names_df1[["name", "confidence", "discard"]].to_dict(orient="index")

        upload_obj(
            names_df1,
            bucket=BUCKET_NAME,
            path_to=f"outputs/topic_names/postproc/level_1.json",
        )

        names_df3 = names_df3.loc[names_df3.index.isin(tax_df["Level_3"].unique())]
        names_df3 = names_df3[["name", "confidence", "discard"]].to_dict(orient="index")

        upload_obj(
            names_df3,
            bucket=BUCKET_NAME,
            path_to=f"outputs/topic_names/postproc/level_3.json",
        )
