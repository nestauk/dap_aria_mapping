import time, pickle, ast, subprocess, re
from dap_aria_mapping import logger, PROJECT_DIR
from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
)
from dap_aria_mapping.utils.topic_names import *
from chatgpt_wrapper import ChatGPT, AsyncChatGPT
from dap_aria_mapping import chatgpt_args
from copy import deepcopy

OUTPUT_DIR = PROJECT_DIR / "outputs" / "interim" / "country_tags"
OUTPUT_DIR_PREPRUNE = PROJECT_DIR / "outputs" / "interim" / "preprune_topics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    try:
        logger.info("Loading tagedd_entities.pkl")
        with open(OUTPUT_DIR / "tagged_entities.pkl", "rb") as f:
            tagged_entities = pickle.load(f)
    except FileNotFoundError:
        logger.info("No tagged_entities.pkl found, creating new list.")
        tagged_entities = []

    logger.info("Loading entities")
    with open(OUTPUT_DIR_PREPRUNE / "discarded_entities.pkl", "rb") as f:
        discarded_entities = pickle.load(f)
    logger.info(f"Loaded prepruned entities - Length: {len(discarded_entities)}")
    entities_to_check = deepcopy(discarded_entities)

    # Prompt chatGPT with the problem
    bot = ChatGPT(False, timeout=360)
    response = bot.ask(chatgpt_args["TAG-INTRO"])
    logger.info(f"Initial response: {response}")

    time.sleep(np.random.uniform(5, 20))

    # Iteratively remove entities and prompt chatGPT
    failed_attempts = 0
    while len(entities_to_check) > 0:
        if len(entities_to_check) >= 100:
            # Get a sample of entities
            entity_sample = random.sample(entities_to_check, 100)
        else:
            entity_sample = entities_to_check

        print(entity_sample)

        logger.info("Asking model for response...")
        response_ok, response, response_reason = bot.ask(f"{entity_sample}")
        time.sleep(np.random.uniform(1, 10))

        # Attempt to convert to list, if exception, try making it explicit
        try:
            print(response)
            # Deal with apostrophes within strings, and excess strings
            start = response.find("[('")
            end = response.rfind("'])]")
            response = response[start : end + 4]
            response = re.sub(r"(\w)'(\w|\s)", r"\1\'\2", response)

            response = ast.literal_eval(response)
        except Exception as e:
            logger.info(
                f"FAILURE: {response_reason} - ChatGPT response is not a list: {response}."
            )
            time.sleep(np.random.uniform(15, 30))
            response_ok, response, response_reason = bot.ask(
                chatgpt_args["TAG-ERROR-NOLIST"]
            )
            print("Before:" + str(response))
            try:
                # response = re.findall(r"\[.*\]", response)[0]
                start = response.find("[('")
                end = response.rfind("'])]")
                response = response[start : end + 4]
                response = re.sub(r"(\w)'(\w|\s)", r"\1\'\2", response)
                print("After:" + str(response))
                response = ast.literal_eval(response)
            except Exception as e:
                logger.info(
                    f"FAILURE {response_reason} - ChatGPT response is not a list: {response}. Skipping"
                )
                failed_attempts += 1
                subprocess.run("pkill firefox", shell=True)
                time.sleep(np.random.uniform(45, 60))
                bot = ChatGPT(False, timeout=360)
                time.sleep(np.random.uniform(10, 20))
                if failed_attempts >= 5:
                    logger.info("Too many failed attempts, idling.")
                    subprocess.run("pkill firefox", shell=True)
                    time.sleep(np.random.randint(300, 600))
                    bot = ChatGPT(False, timeout=360)
                    time.sleep(np.random.uniform(10, 20))
                continue

        # If successful, add to tagged_entities and remove from entities
        logger.info("SUCCESS - ChatGPT response is a list: {}".format(response))
        tagged_entities += response
        entities_to_check = [e for e in entities_to_check if e not in entity_sample]
        failed_attempts = 0

        # Save progress
        logger.info("Saving progress")
        with open(OUTPUT_DIR / "tagged_entities.pkl", "wb") as f:
            pickle.dump(tagged_entities, f)

        # Wait before next iteration
        time.sleep(np.random.uniform(60, 90))
