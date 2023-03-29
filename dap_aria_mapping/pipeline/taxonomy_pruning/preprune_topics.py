import time, pickle, ast, subprocess, re
from dap_aria_mapping import logger, PROJECT_DIR
from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
)
from dap_aria_mapping.utils.topic_names import *
from chatgpt_wrapper import ChatGPT, AsyncChatGPT
from dap_aria_mapping import chatgpt_args

OUTPUT_DIR = PROJECT_DIR / "outputs" / "interim" / "preprune_topics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    try:
        logger.info("Loading preprune_topics.pkl")
        with open(OUTPUT_DIR / "preprune_topics.pkl", "rb") as f:
            prepruned_entities = pickle.load(f)
    except FileNotFoundError:
        logger.info("No preprune_topics.pkl found, creating new list.")
        prepruned_entities = []

    try:
        logger.info("Loading entities")
        with open(OUTPUT_DIR / "entities.pkl", "rb") as f:
            entities = pickle.load(f)
        logger.info(f"Loaded entities - Length: {len(entities)}")
    except FileNotFoundError:
        logger.info("No entities.pkl found, creating new list.")
        entities = get_cooccurrence_taxonomy().index.tolist()
        entities = entities[:35000]

    # Prompt chatGPT with the problem
    bot = ChatGPT(False)
    initial_script = f"I have created a hierarchical taxonomy of topics that are relevant to the academic literature \
        and the commercialization of science. This taxonomy is created from a list of entities, which \
        represent Wikipedia topics and which should represent relevant concepts in academia. These entities \
        are clustered into topics given a co-occurrence matrix. \n\n \
        The problem is that some of these entities may instead represent modern day countries, regions, or cities. \
        If possible, also identify institutions and films. \
        \n\n \
        Your task is to identify these entities and remove them from the taxonomy. \
        I will give you a list of entities and you will return the list of entities to remove. \
        \n\n \
        Your response should be a single Python list of entities that should be removed. \
        For example, if you think that the entities 'United States', 'United Kingdom', and 'Canada' \
        should be removed, you would respond with ['United States', 'United Kingdom', 'Canada']. \
        \n\n \
        If you think that no entities should be removed, you would respond with an empty list, \
        like this: []. \n\n \
        Here is a first list of entities: \
        \n\n \
        {entities[:50]} \n\n"

    response = bot.ask(initial_script)
    logger.info(f"Initial response: {response}")

    time.sleep(np.random.uniform(5, 20))
    bot.ask(
        "I will proceed to give you lists of entities, and you will return the list of entities to remove."
    )

    # Iteratively remove entities and prompt chatGPT
    failed_attempts = 0
    while len(entities) > 0:
        if len(entities) >= 100:
            # Get a sample of entities
            entity_sample = random.sample(entities, 100)
        else:
            entity_sample = entities

        # Define script & get response
        script = f"Your task is to identify entities in a list that represent modern day countries, regions, or cities. \
            If possible, also identify institutions and films. \
            \n\n \
            Your response should be a single Python list of entities that correspond to these categories. Please return a single list. \
            For example, if you think that the entities 'United States', 'United Kingdom', and 'Canada' \
            correspond to modern-day countries, you would respond with ['United States', 'United Kingdom', 'Canada']. \
            \n\n \
            If you think that no entities match any of the above categories, you would respond with an empty list, \
            like this: []. \n\n \
            Here is the list of entities: \
            \n\n \
            {entity_sample} \n\n"

        logger.info("Asking model for response...")
        response_ok, response, response_reason = bot.ask(script)
        time.sleep(np.random.uniform(1, 10))

        # Attempt to convert to list, if exception, try making it explicit
        try:
            print(response)
            response = re.findall("\[.*\]", response)
            if len(response) > 1:
                time.sleep(np.random.uniform(15, 30))
                response_ok, response, response_reason = bot.ask(
                    chatgpt_args["PRUNE-ERROR"]
                )
                response = re.findall("\[.*\]", response)[0]
            else:
                response = response[0]
            response = re.sub("(?<!,)'(?!\s)", '"', response)
            response = ast.literal_eval(response)
        except Exception as e:
            logger.info(
                f"FAILURE: {response_reason} - ChatGPT response is not a list: {response}."
            )
            time.sleep(np.random.uniform(15, 30))
            response_ok, response, response_reason = bot.ask(
                chatgpt_args["PRUNE-ERROR"]
            )
            print("Before:" + str(response))
            try:
                try:  # There is a list somewghere in the response
                    if len(response) > 1:
                        time.sleep(np.random.uniform(15, 30))
                        response_ok, response, response_reason = bot.ask(
                            chatgpt_args["PRUNE-ERROR"]
                        )
                        response = re.findall("\[.*\]", response)[0]
                    else:
                        response = response[0]
                    found_list = True
                except:  # The list is itemized and broken into lines. Identify the line breaks,
                    # and then join the lines into a single list
                    try:
                        lines = response.split("\n")
                        index = None
                        for i, line in enumerate(lines):
                            if line.startswith("- "):
                                index = i
                                break
                        if index is not None:
                            response = "\n".join(lines[index:])
                            response = (
                                "[" + str.replace("\n", ", ").replace(" - ", "") + "]"
                            )
                    except:
                        pass
                print("After:" + str(response))
                response = re.sub("(?<!,)'(?!\s)", '"', response)
                response = ast.literal_eval(response)
            except Exception as e:
                logger.info(
                    f"FAILURE {response_reason} - ChatGPT response is not a list: {response}. Skipping"
                )
                failed_attempts += 1
                subprocess.run("pkill firefox", shell=True)
                time.sleep(np.random.uniform(45, 60))
                bot = ChatGPT(False)
                time.sleep(np.random.uniform(10, 20))
                if failed_attempts >= 5:
                    logger.info("Too many failed attempts, idling.")
                    subprocess.run("pkill firefox", shell=True)
                    time.sleep(np.random.randint(300, 600))
                    bot = ChatGPT(False)
                    time.sleep(np.random.uniform(10, 20))
                continue

        # If successful, add to prepruned_entities and remove from entities
        logger.info("SUCCESS - ChatGPT response is a list: {}".format(response))
        logger.info(f"Non-pruned entities: {len(entities)}")
        prepruned_entities += response
        entities = [e for e in entities if e not in entity_sample]
        failed_attempts = 0

        # Save progress
        logger.info("Saving progress")
        with open(OUTPUT_DIR / "preprune_topics.pkl", "wb") as f:
            pickle.dump(prepruned_entities, f)
        with open(OUTPUT_DIR / "entities.pkl", "wb") as f:
            pickle.dump(entities, f)

        # Wait before next iteration
        time.sleep(np.random.uniform(60, 90))
