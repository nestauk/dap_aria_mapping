"""This script makes topic name assignments for a series of taxonomy levels.
    It can be run from the command line with the following arguments:
        - taxonomy: The type of taxonomy to use. Can be a single taxonomy or a sequence
            of taxonomies, and accepts 'cooccur', 'centroids' or 'imbalanced' as tags.
        - name_type: The type of names to use. Can be 'entity', 'journal', or both.
        - level: The level of the taxonomy to use. Can be a single level or a sequence
        - n_top: The number of top elements to show. In the case of 'chatgpt', this is the
            number of entities to use to label the topic.
        - n_articles: The number of articles to use to count entities in a topic.
            -1 defaults to all.
        - save: Whether to save the topic names to S3.
        - show_count: Whether to show the counts of the entities in each topic. Used
            in name_type 'chatgpt' as part of the query.

Raises:
    Exception: If the taxonomy is not one of 'cooccur', 'semantic' or 'semantic_kmeans'.

Returns:
    json: A json file containing the topic names, saved to S3.
"""
import argparse, json, boto3, time, subprocess
from toolz import pipe
from functools import partial
from nesta_ds_utils.loading_saving.S3 import get_bucket_filenames_s3
from dap_aria_mapping import logger, PROJECT_DIR, BUCKET_NAME, chatgpt_args
from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
    get_semantic_taxonomy,
    get_topic_names,
)
from nesta_ds_utils.loading_saving.S3 import upload_obj
from dap_aria_mapping.getters.openalex import get_openalex_works, get_openalex_entities
from dap_aria_mapping.utils.entity_selection import get_sample, filter_entities
from dap_aria_mapping.utils.topic_names import *
from dap_aria_mapping.utils.chatgpt import revChatGPTWrapper, webChatGPTWrapper


def save_names(
    taxonomy: str, name_type: str, level: int, n_top: int, names: Dict
) -> None:
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
            of taxonomies, and accepts 'cooccur', 'centroids', 'imbalanced', \
            'revChatGPT', or 'webChatGPT' as tags."
        ),
        required=True,
    )

    parser.add_argument(
        "--name_type",
        nargs="+",
        help="The type of names to use. Can be 'entity', 'journal', revChatGPT, or webChatGPT.",
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

    parser.add_argument(
        "--postproc",
        help="Whether to use the postprocessed chatgpt names.",
        action="store_true",
    )

    parser.add_argument("--save", help="Whether to save the plot.", action="store_true")

    parser.add_argument(
        "--show_count",
        help="Whether to show counts for entity names.",
        action="store_true",
    )

    args = parser.parse_args()
    bucket = boto3.resource("s3").Bucket(BUCKET_NAME)

    logger.info("Loading data - taxonomy")
    taxonomies = []
    if "cooccur" in args.taxonomy:
        cooccur_taxonomy = get_cooccurrence_taxonomy(postproc=args.postproc)
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
            partial(
                filter_entities, min_freq=10, max_freq=1_000_000, method="absolute"
            ),
        )

        logger.info("Building dictionary - journal to entities")
        journal_entities = get_journal_entities(oa_works, oa_entities)

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
                    if args.postproc:
                        name_type = "entity_postproc"
                    else:
                        name_type = "entity"
                    logger.info("Saving dictionary as pickle")
                    save_names(
                        taxonomy=taxlabel,
                        name_type=name_type,
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

            if "chatgpt" in args.name_type[0].lower():
                logger.info("Loading ChatGPT names")
                if args.postproc:
                    entity_names = get_topic_names(
                        taxonomy_class=taxlabel,
                        name_type="entity_postproc",
                        level=level,
                        n_top=args.n_top,
                    )
                else:
                    entity_names = get_topic_names(
                        taxonomy_class=taxlabel,
                        name_type="entity",
                        level=level,
                        n_top=args.n_top,
                    )

                if not args.postproc:
                    files_with_name = get_bucket_filenames_s3(
                        bucket_name=BUCKET_NAME,
                        dir_name=f"outputs/topic_names/class_{taxlabel}_nametype_chatgpt_top_{args.n_top}_level_{level}.json",
                    )
                else:
                    files_with_name = get_bucket_filenames_s3(
                        bucket_name=BUCKET_NAME,
                        dir_name=f"outputs/topic_names/postproc/level_{level}.json",
                    )

                if len(files_with_name) > 0:
                    logger.info("ChatGPT names already exist")
                    chatgpt_names = get_topic_names(
                        taxonomy_class=taxlabel,
                        name_type="chatgpt",
                        level=level,
                        n_top=args.n_top,
                        postproc=args.postproc,
                    )

                    num_topics_file, num_topics_total = (
                        len(
                            set(chatgpt_names.keys()).intersection(
                                set(entity_names.keys())
                            )
                        ),
                        len(entity_names.keys()),
                    )

                    if num_topics_file == num_topics_total:
                        logger.info(
                            "ChatGPT names already exist for all topics. Skipping."
                        )
                        continue
                    else:
                        logger.info(
                            f"ChatGPT names already exist for {num_topics_file} out of {num_topics_total} topics"
                        )
                        entity_names = {
                            k: v
                            for k, v in entity_names.items()
                            if k not in chatgpt_names.keys()
                        }
                    first_parse = False

                else:
                    logger.info("ChatGPT names do not exist")
                    chatgpt_names = defaultdict(dict)
                    num_topics_file, num_topics_total, first_parse = (
                        0,
                        len(entity_names.keys()),
                        True,
                    )

                if args.name_type[0] == "revChatGPT":
                    revchatgpt = revChatGPTWrapper(
                        first_parse=first_parse,
                        logger=logger,
                        taxlabel=taxlabel,
                        level=level,
                        args=args,
                    )
                elif args.name_type[0] == "webChatGPT":
                    webchatgpt = webChatGPTWrapper(
                        first_parse=first_parse,
                        logger=logger,
                        taxlabel=taxlabel,
                        level=level,
                        args=args,
                    )

                while num_topics_file < num_topics_total:
                    logger.info(f"Selecting random chatbot to use")

                    if args.name_type[0] == "revChatGPT":
                        chatbot_num = np.random.randint(
                            1, len(chatgpt_args["TOKENS"]) + 1
                        )
                        chatbot_num = 1
                        logger.info(
                            f"Starting batch of topics using chatbot {chatbot_num}"
                        )
                    try:
                        random_sample = random.sample(list(entity_names.keys()), 6)
                    except:
                        random_sample = entity_names.keys()
                    sample_entities = [
                        (topic, entity_names[topic]) for topic in random_sample
                    ]

                    chunk_str = "\n\n ".join(
                        ["List " + ': "'.join(x) + '"' for x in sample_entities]
                    )

                    first_query = (
                        chatgpt_args["QUESTION"]
                        + " \n\n "
                        + f"{chunk_str}"
                        + " \n\n "
                        + chatgpt_args["CLARIFICATION"]
                        + " \n\n"
                        + chatgpt_args["EXAMPLE"]
                        + " \n\n"
                        + chatgpt_args["REQUEST"]
                    )

                    routine_query = (
                        chatgpt_args["NEXT"]
                        + f"\n\n {chunk_str} \n\n"
                        + chatgpt_args["CLARIFICATION"]
                        + "\n\n"
                        + chatgpt_args["EXAMPLE"]
                    )

                    error_query = (
                        chatgpt_args["ERROR"]
                        + "\n\n"
                        + chatgpt_args["EXAMPLE"]
                        + "\n\n Please try again. Only return the list in the required structure. \n\n"
                    )

                    tries = 0
                    while True:
                        logger.info("Asking ChatGPT")
                        time.sleep(np.random.uniform(1, 3))
                        try:
                            if args.name_type[0] == "revChatGPT":
                                chatgpt_names = revchatgpt(
                                    chatbot_num=chatbot_num,
                                    chatgpt_names=chatgpt_names,
                                    first_query=first_query,
                                    routine_query=routine_query,
                                    error_query=error_query,
                                )
                            elif args.name_type[0] == "webChatGPT":
                                chatgpt_names = webchatgpt(
                                    chatgpt_names=chatgpt_names,
                                    tries=tries,
                                    first_query=first_query,
                                    routine_query=routine_query,
                                    error_query=error_query,
                                )

                            # Update number of topics with names & pending groups
                            entity_names = {
                                k: v
                                for k, v in entity_names.items()
                                if k not in chatgpt_names.keys()
                            }
                            if args.save:
                                logger.info("Saving dictionary as json")
                                if not args.postproc:
                                    save_names(
                                        taxonomy=taxlabel,
                                        name_type="chatgpt",
                                        level=level,
                                        n_top=args.n_top,
                                        names=chatgpt_names,
                                    )
                                else:
                                    upload_obj(
                                        chatgpt_names,
                                        bucket=BUCKET_NAME,
                                        path_to=f"outputs/topic_names/postproc/level_{str(level)}.json",
                                    )
                            break

                        except Exception as e:
                            tries += 1
                            logger.info(
                                f"ChatGPT failed to respond. Reason: {e}. Trying again."
                            )
                            time.sleep(np.random.randint(6, 12))
                            if tries > 3:
                                logger.info(
                                    "ChatGPT failed to respond. Idling for 5-10 minutes."
                                )
                                # pkill firefox using subprocess
                                subprocess.run("pkill firefox", shell=True)
                                time.sleep(np.random.randint(300, 600))

                                logger.info("Restarting chatbot")
                                webchatgpt = webChatGPTWrapper(
                                    first_parse=first_parse,
                                    logger=logger,
                                    taxlabel=taxlabel,
                                    level=level,
                                    args=args,
                                )
                                break

            logger.info("Finished level {}".format(str(level)))
        logger.info("Finished taxonomy {}".format(taxlabel))
