import pandas as pd
import argparse, json, boto3, time, os, ast
from toolz import pipe
from dap_aria_mapping import logger, PROJECT_DIR, BUCKET_NAME
from functools import partial
from itertools import islice
from chatgpt_wrapper import ChatGPT, AsyncChatGPT
from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
    get_semantic_taxonomy,
    get_topic_names
)
from dap_aria_mapping.getters.openalex import get_openalex_works, get_openalex_entities
from dap_aria_mapping.utils.entity_selection import get_sample, filter_entities
from dap_aria_mapping.utils.topic_names import *

OUTPUT_DIR = PROJECT_DIR / "outputs" / "interim" / "topic_names"

def chunked(it, size):
    it = iter(it)
    while True:
        p = tuple(islice(it, size))
        if not p:
            break
        yield p


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

    parser.add_argument("--show_count", help="Whether to show counts for entity names.", action="store_true")

    args = parser.parse_args()
    bucket = boto3.resource("s3").Bucket(BUCKET_NAME)

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
                names = get_cluster_entity_names(
                    clust_counts, num_entities=args.n_top, show_count=args.show_count
                )
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

                files_with_name = [
                    file.key
                    for file in bucket.objects.filter(
                        Prefix=f"outputs/topic_names/class_{taxlabel}_nametype_chatgpt_top_{args.n_top}_level_{level}.json"
                    )
                ]

                if len(files_with_name) > 0:
                    logger.info("ChatGPT names already exist")
                    chatgpt_names = get_topic_names(
                        taxonomy_class=taxlabel,
                        name_type="chatgpt",
                        level=level,
                        long=False,
                        n_top=args.n_top,
                    )

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
                    logger.info("ChatGPT names do not exist")
                    chatgpt_names = defaultdict(dict)
                    num_topics_file = 0

                first = True
                while num_topics_file < num_topics_total:
                    logger.info(f"Starting batch of topics")
                    random_sample = random.sample(list(entity_names.keys()), 6)
                    sample_entities = [(topic, entity_names[topic]) for topic in random_sample]
                    tries = 0
                    response_ok = False
                    while not response_ok:
                        logger.info("Asking ChatGPT")
                        time.sleep(np.random.uniform(1, 3))
                        try:
                            chunk_str = "\n\n ".join(["List " + ": \"".join(x) + "\"" for x in sample_entities])

                            if first:
                                query = (
                                    f"What are the Wikipedia topics that best describe the following groups of entities (the number of times in parenthesis corresponds to how often these entities are found in the topic, and should be taken into account when making a decision)?" \
                                    " \n\n " \
                                    f"{chunk_str}" \
                                    " \n\n " \
                                    "Please only provide the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, please provide a list of entities that, if discarded, would help identify a topic. The structure of the answer should be a list of tuples of four elements: (list identifier, topic name, confidence score, list of entities to discard (None if there are none)). For example:" \
                                    " \n\n" \
                                    "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]" \
                                    " \n\n" \
                                    "Please avoid very general topic names (such as 'Science' or 'Technology') and return only the list of tuples with the answers using the structure above (it will be parsed by Python's ast literal_eval method)." \
                                )
                                response = bot.ask(query)
                            else:
                                query = f"{chunk_str}"
                                response = bot.ask(query)

                            try:
                                response = ast.literal_eval(response)
                                if isinstance(response, list):
                                    logger.info(f"SUCCESS - ChatGPT response: {response}")
                                    for quadtuple in response:
                                        list_id = quadtuple[0].split(" ")[-1]
                                        topic = quadtuple[1]
                                        confidence = quadtuple[2]
                                        discard = quadtuple[3]
                                        chatgpt_names[list_id] = {"name": topic, "confidence": confidence, "discard": discard}

                            except Exception as e:
                                logger.info(f"FAILURE - ChatGPT response is not a list: {response}.")
                                sleep_time = np.random.randint(45, 60)
                                logger.info(f"Routine idling - Sleeping for {sleep_time} seconds")
                                time.sleep(sleep_time)
                                bot.ask(
                                    "The response is not a list. Please output a list of four-item tuples as your answer, as in the following example: \n\n" \
                                    "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]" \
                                )
                                raise Exception("ChatGPT response is not a list.")

                            # Accept response, refresh session, set to skip problem description
                            response_ok = True
                            bot.refresh_session()
                            if first:
                                first = False

                            # Update number of topics with names & pending groups 
                            num_topics_file = len(chatgpt_names.keys())
                            entity_names = {k: v for k, v in entity_names.items() if k not in chatgpt_names.keys()}

                            sleep_time = np.random.randint(90, 120)
                            logger.info(f"Routine idling - Sleeping for {sleep_time} seconds")
                            time.sleep(sleep_time)
                            
                            # load back (in case other streams have updated the dictionary)
                            chatgpt_names_new = get_topic_names(
                                taxonomy_class=taxlabel,
                                name_type="chatgpt",
                                level=level,
                                long=False,
                                n_top=args.n_top,
                            )

                            # merge any missing keys
                            chatgpt_names = {**chatgpt_names, **chatgpt_names_new}

                            if args.save:
                                logger.info("Saving dictionary as json")
                                save_names(
                                    taxonomy=taxlabel,
                                    name_type="chatgpt",
                                    level=level,
                                    n_top=args.n_top,
                                    names=chatgpt_names,
                                )
                            break

                        except Exception as e:
                            tries += 1
                            logger.info(f"ChatGPT failed to respond. Reason: {e}. Trying again.")
                            time.sleep(np.random.randint(6, 12))
                            bot.refresh_session()
                            if tries > 3:
                                logger.info("ChatGPT failed to respond. Idling for 10-12 minutes.")
                                time.sleep(np.random.randint(600, 720))

                    

                # if os.path.exists(OUTPUT_DIR / f"class_{taxlabel}_nametype_chatgpt_top_{args.n_top}_level_{level}.json"):
                #     logger.info("ChatGPT names already exist")
                #     with open(OUTPUT_DIR / f"class_{taxlabel}_nametype_chatgpt_top_{args.n_top}_level_{level}.json", "r") as f:
                #         chatgpt_names = json.load(f)
                #     num_topics_file, num_topics_total = (
                #         len(set(chatgpt_names.keys()).intersection(set(entity_names.keys()))), 
                #         len(entity_names.keys())
                #     )
                #     if num_topics_file == num_topics_total:
                #         logger.info("ChatGPT names already exist for all topics. Skipping.")
                #         continue
                #     else:
                #         logger.info(f"ChatGPT names already exist for {num_topics_file} out of {num_topics_total} topics")
                #         entity_names = {k: v for k, v in entity_names.items() if k not in chatgpt_names.keys()}
                # else:
                #     chatgpt_names = defaultdict(dict)

                # for clust, names in entity_names.items():
                # for chunk in chunked(entity_names.items(), 6):
                #     tries = 0
                #     response_ok = False
                #     first = True
                #     while not response_ok:
                #         sleep_time = np.random.randint(90, 120)
                #         logger.info(f"Routine idling - Sleeping for {sleep_time} seconds")
                #         time.sleep(sleep_time)

                #         try:
                #             chunk_str = "\n\n ".join(["List " + ": \"".join(x) + "\"" for x in chunk] )
                #             if first:
                #                 query = (
                #                     f"What are the Wikipedia topics that best describe the following groups of entities (the number of times in parenthesis corresponds to how often these entities are found in the topic, and should be taken into account when making a decision)?" \
                #                     " \n\n " \
                #                     f"{chunk_str}" \
                #                     " \n\n " \
                #                     "Please only provide the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, please provide a list of entities that, if discarded, would help identify a topic. The structure of the answer should be a list of tuples of four elements: (list identifier, topic name, confidence score, list of entities to discard (None if there are none)). For example:" \
                #                     # "Please only provide the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, please provide a list of entities that, if discarded, would help identify a topic. Also provide a topic name for the discarded entities, if any. The structure of the answer should be a list of tuples of five elements: (list identifier, topic name, confidence score, list of entities to discard (return an empty list if there are none), discarded topic name (None if there are no entities to discard)). For example: " \
                #                     " \n\n" \
                #                     "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]" \
                #                     # "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Russian doll'], 'Russian Elements'), ('List 2', 'Cosmology', 90, ['Berlin', 'Madrid'], 'European cities')]"
                #                     " \n\n" \
                #                     "Please avoid very general topic names (such as 'Science' or 'Technology') and return only the list of tuples with the answers using the structure above." \
                #                     # "In addition, if possible try to avoid topics that are too general - such as 'Science' or 'Technology' - unless a more concrete topic reduces the confidence score significantly." \
                #                     # "Do not explain the reasoning behind the reply, nor write anything else than the answer (ie. do not write notes on why you discard entities)."
                #                 )
                #                 response = bot.ask(query)
                #             else:
                #                 query = f"{chunk_str}"
                #                 response = bot.ask(query)

                #             try:
                #                 response = ast.literal_eval(response)
                #             except Exception as e:
                #                 logger.info(f"FAILURE - ChatGPT response is not a list: {response}.")
                #                 time.sleep(np.random.randint(9, 12))
                #                 bot.ask(
                #                     "The response is not a list. Please output a list of four-item tuples as your answer, as in the following example: \n\n" \
                #                     "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]" \
                #                 )
                #                 raise Exception("ChatGPT response is not a list.")

                #             #     f"What is the Wikipedia topic that best describes the following group of entities?" \
                #             #     f"{names}" \
                #             #     "\n\n" \
                #             #     f"Please only provide the topic name that best describes the group of entities, and a " \
                #             #     f"confidence score between 0 and 100 on how sure you are about the answer." \
                #             #     "The structure of the answer should be: topic name (confidence score)." \
                #             #     "For example: 'Machine learning (100)'. In addition, try to avoid topics that are too " \
                #             #     "general, such as 'Science' or 'Technology'."
                #             # )
                #             if "Unusable response produced" in response:
                #                 logger.info(f"FAILURE - ChatGPT failed to provide an answer: {response}.")
                #                 raise Exception("Unusable response produced. Consider reducing number of requests.")
                #             else:
                #                 logger.info(f"SUCCESS - ChatGPT response: {response}")
                #                 for quadtuple in response:
                #                     list_id = quadtuple[0].split(" ")[-1]
                #                     topic = quadtuple[1]
                #                     confidence = quadtuple[2]
                #                     discard = quadtuple[3]
                #                     chatgpt_names[list_id] = {"name": topic, "confidence": confidence, "discard": discard}
                #                 # chatgpt_names[clust] = response
                #             # logger.info(f"ChatGPT response: {chatgpt_names[clust]}")
                #             with open(OUTPUT_DIR / f"class_{taxlabel}_nametype_chatgpt_top_{args.n_top}_level_{level}.json", "w") as f:
                #                 json.dump(chatgpt_names, f)
                #             bot.refresh_session()
                #             response_ok = True
                #             if first:
                #                 first = False
                #             break

                #         except Exception as e:
                #             tries += 1
                #             logger.info(f"ChatGPT failed to respond. Reason: {e}. Trying again.")
                #             time.sleep(np.random.randint(3, 6))
                #             bot.refresh_session()
                #             if tries > 3:
                #                 logger.info("ChatGPT failed to respond. Idling for 10-12 minutes.")
                #                 # chatgpt_names[clust] = "ChatGPT failed to respond."
                #                 time.sleep(np.random.randint(600, 720))

                # if args.save:
                #     logger.info("Saving dictionary as pickle")
                #     save_names(
                #         taxonomy=taxlabel,
                #         name_type="chatgpt",
                #         level=level,
                #         n_top=args.n_top,
                #         names=chatgpt_names,
                #     )

            logger.info("Finished level {}".format(str(level)))
        logger.info("Finished taxonomy {}".format(taxlabel))
