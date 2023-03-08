import pandas as pd
import argparse, json, boto3, time, os
from toolz import pipe
from dap_aria_mapping import logger, PROJECT_DIR, BUCKET_NAME
from functools import partial
from chatgpt_wrapper import ChatGPT
from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
    get_semantic_taxonomy,
    get_topic_names
)
from dap_aria_mapping.getters.openalex import get_openalex_works, get_openalex_entities
from dap_aria_mapping.utils.entity_selection import get_sample, filter_entities
from dap_aria_mapping.utils.topic_names import *

OUTPUT_DIR = PROJECT_DIR / "outputs" / "interim" / "topic_names"


def save_names(taxonomy: str, name_type: str, level: int, n_top: int, names: Dict) -> None:
    """Save the topic names for a given taxonomy level.

    Args:
        taxonomy (str): A string representing the taxonomy class.
            Can be 'cooccur', 'semantic' or 'semantic_kmeans'.
        name_type (str): A string representing the topic names type.
        level (int): The level of the taxonomy to use.
        n_top (int): The number of top elements to show.
        names (Dict): A dictionary of the topic names.
    """
    names = {str(k): v for k, v in names.items()}
    title = "outputs/topic_names/class_{}_nametype_{}_top_{}_level_{}.json".format(
        taxonomy, name_type, n_top, str(level)
    )

    s3 = boto3.client("s3")
    s3.put_object(
        Body=json.dumps(names),
        Bucket=BUCKET_NAME,
        Key=title,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Make histograms of a given taxonomy level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--taxonomy",
        nargs="+",
        help=(
            "The type of taxonomy to use. Can be a single taxonomy or a sequence \
            of taxonomies, and accepts 'cooccur', 'centroids' or 'imbalanced' as tags."
        ),
        required=True,
    )

    parser.add_argument(
        "--name_type",
        nargs="+",
        help="The type of names to use. Can be 'entity', 'journal', or both.",
        default="entity",
    )

    parser.add_argument(
        "--n_top",
        type=int,
        help="The number of top elements to show. Defaults to 3.",
        default=3,
    )

    parser.add_argument(
        "--levels", nargs="+", help="The levels of the taxonomy to use.", required=True
    )

    parser.add_argument(
        "--n_articles",
        type=int,
        help="The number of articles to use. If -1, all articles are used.",
    )

    parser.add_argument("--save", help="Whether to save the plot.", action="store_true")

    args = parser.parse_args()

    logger.info("Loading data - taxonomy")
    taxonomies = []
    if "cooccur" in args.taxonomy:
        cooccur_taxonomy = get_cooccurrence_taxonomy()
        taxonomies.append(["cooccur", cooccur_taxonomy])
    if "centroids" in args.taxonomy:
        semantic_centroids_taxonomy = get_semantic_taxonomy("centroids")
        taxonomies.append(["centroids", semantic_centroids_taxonomy])
    if "imbalanced" in args.taxonomy:
        semantic_kmeans_taxonomy = get_semantic_taxonomy("imbalanced")
        taxonomies.append(["imbalanced", semantic_kmeans_taxonomy])

    if any(["entity" in args.name_type, "journal" in args.name_type]):
        logger.info("Loading data - works")
        oa_works = get_openalex_works()

        logger.info("Loading data - entities")
        oa_entities = pipe(
            get_openalex_entities(),
            partial(get_sample, score_threshold=80, num_articles=args.n_articles),
            partial(filter_entities, min_freq=10, max_freq=1_000_000, method="absolute"),
        )

        logger.info("Building dictionary - journal to entities")
        journal_entities = get_journal_entities(oa_works, oa_entities)

    if "chatgpt" in args.name_type:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        bot = ChatGPT(False)

    for taxlabel, taxonomy in taxonomies:
        logger.info("Starting taxonomy {}".format(taxlabel))
        for level in [int(x) for x in args.levels]:
            logger.info("Starting level {}".format(str(level)))
            logger.info("Building dictionary - entity to journals")
            if any(["entity" in args.name_type, "journal" in args.name_type]):
                clust_counts = get_level_entity_counts(
                    journal_entities, taxonomy, level=level
                )

            if "entity" in args.name_type:
                logger.info("Building dictionary - entity to names")
                names = get_cluster_entity_names(clust_counts, num_entities=args.n_top)
                if args.save:
                    logger.info("Saving dictionary as pickle")
                    save_names(
                        taxonomy=taxlabel,
                        name_type="entity",
                        level=level,
                        n_top=args.n_top,
                        names=names,
                    )

            if "journal" in args.name_type:
                logger.info("Building dictionary - journal to names")
                cluster_journal_ranks = get_cluster_journal_ranks(
                    clust_counts, journal_entities, output="relative"
                )
                names = get_cluster_journal_names(
                    cluster_journal_ranks, num_names=args.n_top
                )
                if args.save:
                    logger.info("Saving dictionary as pickle")
                    save_names(
                        taxonomy=taxlabel,
                        name_type="journal",
                        level=level,
                        n_top=args.n_top,
                        names=names,
                    )

            if "chatgpt" in args.name_type:
                entity_names = get_topic_names(
                    taxonomy_class=taxlabel, 
                    name_type="entity",
                    level=level, 
                    long=True,
                    n_top=args.n_top,
                )
                if os.path.exists(OUTPUT_DIR / f"class_{taxlabel}_nametype_chatgpt_top_{args.n_top}_level_{level}.json"):
                    logger.info("ChatGPT names already exist")
                    with open(OUTPUT_DIR / f"class_{taxlabel}_nametype_chatgpt_top_{args.n_top}_level_{level}.json", "r") as f:
                        chatgpt_names = json.load(f)
                    num_topics_file, num_topics_total = (
                        len(set(chatgpt_names.keys()).intersection(set(entity_names.keys()))), 
                        len(entity_names.keys())
                    )
                    if num_topics_file == num_topics_total:
                        logger.info("ChatGPT names already exist for all topics. Skipping.")
                        continue
                    else:
                        logger.info(f"ChatGPT names already exist for {num_topics_file} out of {num_topics_total} topics")
                        entity_names = {k: v for k, v in entity_names.items() if k not in chatgpt_names.keys()}
                else:
                    chatgpt_names = {}

                for clust, names in entity_names.items():
                    tries = 0
                    response_ok = False
                    while not response_ok:
                        time.sleep(np.random.randint(45, 90))

                        try:
                            response = bot.ask(
                                f"What is the Wikipedia topic that best describes the following group of entities?" \
                                f"{names}" \
                                "\n\n" \
                                f"Please only provide the topic name that best describes the group of entities, and a " \
                                f"confidence score between 0 and 100 on how sure you are about the answer." \
                                "The structure of the answer should be: topic name (confidence score)." \
                                "For example: 'Machine learning (100)'. In addition, try to avoid topics that are too " \
                                "general, such as 'Science' or 'Technology'."
                            )
                            if "Unusable response produced" in response:
                                logger.info(f"ChatGPT failed to provide an answer: {response}.")
                                raise Exception("Unusable response produced. Consider reducing number of requests.")
                            else:
                                chatgpt_names[clust] = response
                            logger.info(f"ChatGPT response: {chatgpt_names[clust]}")
                            with open(OUTPUT_DIR / f"class_{taxlabel}_nametype_chatgpt_top_{args.n_top}_level_{level}.json", "w") as f:
                                json.dump(chatgpt_names, f)
                            bot.refresh_session()
                            response_ok = True
                            break

                        except:
                            tries += 1
                            logger.info("ChatGPT failed to respond. Trying again.")
                            time.sleep(np.random.randint(3, 6))
                            bot.refresh_session()
                            if tries > 3:
                                logger.info("ChatGPT failed to respond. Idling for 15-20 minutes.")
                                chatgpt_names[clust] = "ChatGPT failed to respond."
                                time.sleep(np.random.randint(900, 1200))

                if args.save:
                    logger.info("Saving dictionary as pickle")
                    save_names(
                        taxonomy=taxlabel,
                        name_type="chatgpt",
                        level=level,
                        n_top=args.n_top,
                        names=chatgpt_names,
                    )

            logger.info("Finished level {}".format(str(level)))
        logger.info("Finished taxonomy {}".format(taxlabel))
