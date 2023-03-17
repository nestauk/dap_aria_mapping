import numpy as np
from typing import Dict, List, Any, Tuple, Sequence
from revChatGPT.V1 import Chatbot
from chatgpt_wrapper import ChatGPT, AsyncChatGPT
from dap_aria_mapping.getters.taxonomies import (
    get_topic_names,
)

import logging, random, time, ast, argparse
from dap_aria_mapping import chatgpt_args


class revChatGPTWrapper:
    def __init__(
        self,
        first_parse: bool,
        logger: logging.Logger,
        taxlabel: str,
        level: int,
        args: argparse.Namespace,
    ):
        self.first_parse = first_parse
        self.logger = logger
        self.taxlabel = taxlabel
        self.level = level
        self.args = args

        self.first = {
            f"chatbot{n}": True for n in range(1, len(chatgpt_args["TOKENS"]) + 1)
        }
        self.chatbots = {
            f"chatbot{i}": Chatbot(config={"session_token": token})
            for i, token in enumerate(chatgpt_args["TOKENS"], 1)
        }

    def __call__(
        self,
        chatbot_num: int,
        chatgpt_names: Dict[str, Dict[str, Any]],
        sample_entities: List[List[str]],
    ) -> Dict[str, Sequence[Tuple[str, str, int, List[str]]]]:

        chatbot = self.chatbots[f"chatbot{chatbot_num}"]
        chunk_str = "\n\n ".join(
            ["List " + ': "'.join(x) + '"' for x in sample_entities]
        )

        if self.first[f"chatbot{chatbot_num}"]:
            for data in chatbot.ask(
                random.choice(["Hi!", "Hello!", "Hey!", "Hi there!"])
            ):
                response = data["message"]

            query = (
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
            for data in chatbot.ask(query):
                response = data["message"]
        else:
            query = (
                chatgpt_args["NEXT"]
                + f"\n\n {chunk_str} \n\n"
                + chatgpt_args["CLARIFICATION"]
                + "\n\n"
                + chatgpt_args["EXAMPLE"]
            )
            for data in chatbot.ask(query):
                response = data["message"]

        # Attempt to convert to list, if exception, try making it explicit
        try:
            response = ast.literal_eval(response)
        except Exception as e:
            self.logger.info(f"FAILURE - ChatGPT response is not a list: {response}.")
            sleep_time = np.random.randint(4, 6)
            self.logger.info(
                f"Your response is not a list with the requested structure. Remember that I only want the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, also provide a list of entities that, if discarded, would help identify a topic. The structure of the answer should be a list of tuples of four elements: [(list identifier, topic name, confidence score, list of entities to discard (None if there are none)), ... ]. For example:utine idling - Sleeping for {sleep_time} seconds"
            )
            time.sleep(sleep_time)
            query = (
                chatgpt_args["ERROR"]
                + "\n\n"
                + chatgpt_args["EXAMPLE"]
                + "\n\n Please try again. \n\n"
            )
            for data in chatbot.ask(query):
                response = data["message"]

        # In case it failed to output a list the first time but not the second
        try:
            if not isinstance(response, list):
                response = ast.literal_eval(response)
        except Exception as e:
            raise Exception(f"ChatGPT response is not a list: {e}")

        self.logger.info(f"SUCCESS - ChatGPT response: {response}")
        for quadtuple in response:
            list_id = quadtuple[0].split(" ")[-1]
            topic = quadtuple[1]
            confidence = quadtuple[2]
            discard = quadtuple[3]
            chatgpt_names[list_id] = {
                "name": topic,
                "confidence": confidence,
                "discard": discard,
            }

        # Accept response, refresh session, set to skip problem description
        if self.first[f"chatbot{chatbot_num}"]:
            self.first[f"chatbot{chatbot_num}"] = False

        sleep_time = np.random.randint(9, 12)
        self.logger.info(f"Routine idling - Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)

        # load back (in case other streams have updated the dictionary)
        if not self.first_parse:
            chatgpt_names_updated = get_topic_names(
                taxonomy_class=self.taxlabel,
                name_type="chatgpt",
                level=self.level,
                n_top=self.args.n_top,
            )
        else:
            chatgpt_names_updated = {}
            self.first_parse = False

        # merge any missing keys
        chatgpt_names = {**chatgpt_names, **chatgpt_names_updated}

        return chatgpt_names


class webChatGPTWrapper:
    def __init__(
        self,
        first_parse: bool,
        logger: logging.Logger,
        taxlabel: str,
        level: int,
        args: argparse.Namespace,
    ):
        self.first_parse = first_parse
        self.logger = logger
        self.taxlabel = taxlabel
        self.level = level
        self.args = args

        self.bot = ChatGPT(False)
        self.first = True

    def __call__(
        self,
        sample_entities: List[List[str]],
        chatgpt_names: Dict[str, Sequence[Tuple[str, str, int, List[str]]]],
        tries: int,
    ) -> Dict[str, Sequence[Tuple[str, str, int, List[str]]]]:

        chunk_str = "\n\n ".join(
            ["List " + ': "'.join(x) + '"' for x in sample_entities]
        )

        if self.first:
            query = (
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
            response = self.bot.ask(query)
        else:
            query = (
                chatgpt_args["NEXT"]
                + f"\n\n {chunk_str} \n\n"
                + chatgpt_args["CLARIFICATION"]
                + "\n\n"
                + chatgpt_args["EXAMPLE"]
            )
            response = self.bot.ask(query)

        # Attempt to convert to list, if exception, try making it explicit
        try:
            response_str = ast.literal_eval(response[1])
        except Exception as e:
            self.logger.info(
                f"FAILURE - ChatGPT response is not a list: {response[1]}. Reason: {response[2]}"
            )
            sleep_time = np.random.randint(40, 60)
            self.logger.info(f"Routine idling - Sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)
            if tries > 2:
                query = (
                    chatgpt_args["ERROR"]
                    + "\n\n"
                    + chatgpt_args["EXAMPLE"]
                    + "\n\n Please try again. Only return the list in the required structure. \n\n"
                )
                response = self.bot.ask(query)

        # In case it failed to output a list the first time but not the second
        try:
            response_str = ast.literal_eval(response[1])
        except Exception as e:
            raise Exception(f"ChatGPT response is not a list: {e}")

        # if past the exception, assume successful response
        self.logger.info(f"SUCCESS - ChatGPT response: {response_str}")
        for quadtuple in response_str:
            list_id = quadtuple[0].split(" ")[-1]
            topic = quadtuple[1]
            confidence = quadtuple[2]
            discard = quadtuple[3]
            chatgpt_names[list_id] = {
                "name": topic,
                "confidence": confidence,
                "discard": discard,
            }

        # Accept response, refresh session, set to skip problem description
        if self.first:
            self.first = False

        sleep_time = np.random.randint(60, 90)
        self.logger.info(f"Routine idling - Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)

        # load back (in case other streams have updated the dictionary)
        if not self.first_parse:
            chatgpt_names_updated = get_topic_names(
                taxonomy_class=self.taxlabel,
                name_type="chatgpt",
                level=self.level,
                n_top=self.args.n_top,
            )
        else:
            chatgpt_names_updated = {}
            self.first_parse = False

        # merge any missing keys
        chatgpt_names = {**chatgpt_names, **chatgpt_names_updated}

        return chatgpt_names
