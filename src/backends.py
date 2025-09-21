"""
This module is responsible for defining the backends used to communicate with
the models.
"""

import base64
from io import BytesIO
import json
import logging
import time
from types import NoneType
from typing import Any, Dict, List, Union

import ollama
import openai

from constants import (
    ERROR_TIMEOUT,
    JUDGE_FORMAT,
    JUDGE_PROMPT,
    MAX_TOKENS_ANSWER,
    MAX_TOKENS_JUDGE,
    SYSTEM_EXACT_ANSWER,
    SYSTEM_MC,
    USE_EXPERIMENTAL_IMAGE_UPLOAD,
)


MessagesType = List[Dict[str, Any]]
SchemaType = Dict[str, Any]

BASE64_IMAGE_URL = "data:image/jpeg;base64,"


class Backend:
    """
    A base class for backends that can be used to query models.
    """

    def __init__(self, endpoint: str, api_key: str, logger: logging.Logger):
        """
        Initializes the backend with the given endpoint and API key.
        """

        self.endpoint = endpoint
        self.api_key = api_key
        self._client = None
        self.logger = logger

    def _query_function(
        self,
        model: str,
        messages: MessagesType,
        num_predict: int,
        format_: Union[SchemaType, NoneType] = None,
    ):
        """
        A function that queries the model with the given messages and returns
        the response. Needs to implement logic for accepting num_predict and
        format for structured outputs.
        """

        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )

    def prompt_model(self, model: str, question: Dict[str, str]) -> str:
        """
        Prompts a given model (which is taking the exam) with a question.
        """

        user_message: "Dict[str, Union[str, List[str]]]" = {
            "role": "user",
            "content": question["question"],
        }

        if question["image"]:
            # attach an image if there is one in the question
            user_message["images"] = [question["image"].split(",")[-1]]

        messages = [
            {
                "role": "system",
                "content": (
                    SYSTEM_EXACT_ANSWER
                    if question["answer_type"] == "exactMatch"
                    else SYSTEM_MC
                ),
            },
            user_message,
        ]

        response = self._query_function(
            model=model,
            messages=messages,
            num_predict=MAX_TOKENS_ANSWER,
        )

        return response

    def prompt_judge_model(
        self,
        model: str,
        question: Dict[str, str],
        model_answer: str,
        correct_answer: str,
    ) -> bool:
        """
        Prompts a judge model with the question, the response and the correct
        answer and lets it decide whether the response is correct or not.
        """

        while True:
            # (as long as the judge model is unsure and hasn't given a
            # definitive answer yet)

            try:
                response = self._query_function(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": JUDGE_PROMPT.format(
                                question=question,
                                correct_answer=correct_answer,
                                response=model_answer,
                            ),
                        },
                    ],
                    num_predict=MAX_TOKENS_JUDGE,
                    format_=JUDGE_FORMAT,
                )

                return json.loads(response)["is_answer_correct"]
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.error("Error while judging: %s", e)
                self.logger.info("Retrying in %s seconds...", ERROR_TIMEOUT)

                time.sleep(ERROR_TIMEOUT)


class OllamaBackend(Backend):
    """
    A backend that uses the Ollama API to query models.
    """

    def __init__(self, endpoint: str, api_key: str, logger: logging.Logger):
        """
        Initializes the Ollama backend with the given endpoint and API key.
        """

        super().__init__(endpoint, api_key if api_key else "ollama", logger)

        self._client = ollama.Client(
            host=endpoint,
            headers={"Authentication": f"Bearer {self.api_key}"},
        )

    def _query_function(
        self,
        model: str,
        messages: MessagesType,
        num_predict: int,
        format_: Union[SchemaType, NoneType] = None,
    ) -> str:
        """
        Queries the Ollama model with the given messages and returns the
        response.
        """

        response = self._client.chat(
            model=model,
            messages=messages,
            options={"num_predict": num_predict},
            format=format_,
        )

        if not response or not response["message"]["content"]:
            return ""

        return response["message"]["content"]


class OpenAIBackend(Backend):
    """
    A backend that uses the OpenAI API to query models.
    """

    def __init__(self, endpoint: str, api_key: str, logger: logging.Logger):
        """
        Initializes the OpenAI backend with the given endpoint and API key.
        """

        super().__init__(endpoint, (api_key if api_key else "ollama"), logger)

        self._client = openai.OpenAI(base_url=endpoint, api_key=self.api_key)
        print("openai backend used")

    def _query_function(
        self,
        model: str,
        messages: MessagesType,
        num_predict: int,
        format_: Union[SchemaType, NoneType] = None,
    ) -> str:
        """
        Queries the OpenAI model with the given messages and returns the
        response.
        """

        converted_messages: MessagesType = []

        for message in messages:
            if "images" in message and len(message["images"]) > 0:
                if USE_EXPERIMENTAL_IMAGE_UPLOAD:
                    image_data = base64.b64decode(
                        message["images"][0].split(",", 1)[-1]
                    )
                    image_file = BytesIO(image_data)
                    image_file.name = "image.jpg"
                    file = openai.files.create(
                        file=image_file, purpose="vision"
                    )
                    file_id = file.id

                    converted_messages.append(
                        {
                            "role": message["role"],
                            "content": [
                                {"type": "text", "text": message["content"]},
                                {
                                    "type": "image_file",
                                    "image_file": {"file_id": file_id},
                                },
                            ],
                        }
                    )
                else:
                    url = BASE64_IMAGE_URL + message["images"][0]
                    converted_messages.append(
                        {
                            "role": message["role"],
                            "content": [
                                {"type": "text", "text": message["content"]},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": url},
                                },
                            ],
                        }
                    )

            else:
                converted_messages.append(
                    {"role": message["role"], "content": message["content"]}
                )

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            max_tokens=num_predict,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_response_schema",
                    "schema": format_,
                },
            },  # type: ignore
        )

        if not response or not response.choices[0].message.content:
            return ""

        return response.choices[0].message.content


BACKEND_NAMES = {
    "ollama": OllamaBackend,
    "openai": OpenAIBackend,
}
