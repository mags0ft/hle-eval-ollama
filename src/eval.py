"""
Evaluation script to run "Humanity's Last Exam" on Ollama models.
"""

from email.mime import image
from http import client
import random
from typing import Union
import tqdm
import ollama
import datasets
import json
import dataclasses
import uuid
import argparse
import os

from logger import create_logger


p = argparse.ArgumentParser()
p.add_argument("--model", type=str, required=True)
p.add_argument("--judge", type=str, required=True)
p.add_argument("--dataset", type=str)
p.add_argument("--num-questions", type=int)


STD_DATASET: str = "cais/hle"

# taken from line 11-13,
# https://github.com/centerforaisafety/hle
SYSTEM_EXACT_ANSWER = """Your response should be in the following format:\n\
Explanation: {your explanation for your final answer}\nExact Answer: {your \
succinct, final answer}\nConfidence: {your confidence score between 0% and \
100% for your answer}"""

SYSTEM_MC = """Your response should be in the following format:\nExplanation: \
{your explanation for your answer choice}\nAnswer: {your chosen answer}\n\
Confidence: {your confidence score between 0% and 100% for your answer}"""


# taken from line 16-33,
# https://github.com/centerforaisafety/hle
JUDGE_PROMPT = """Judge whether the following [response] to [question] is \
correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

The following text describes the actual, correct answer to the question.

[correct_answer]: {correct_answer}

After reading the [correct_answer], please judge: is the [response] correct?

Explain why the final answer in [response] is correct or incorrect based on \
[correct_answer], focusing only on if there are meaningful differences \
between [correct_answer] and the extracted final answer. Do not comment on \
any background to the problem, do not attempt to solve the problem, do not \
argue for any answer different than [correct_answer], focus only on whether \
the answers match.

Finally, answer 'YES, CORRECT' only if extracted_final_answer matches the \
[correct_answer] given above, or is within a small margin of error for \
numerical problems. Answer 'NO, INCORRECT' otherwise, i.e. if there if there \
is any inconsistency, ambiguity, non-equivalency, or if the extracted answer \
is incorrect.
"""


@dataclasses.dataclass
class IndividualResult:
    """
    Dataclass for representing an exam result for one individual model.
    """

    correct: int = 0
    wrong: int = 0

    # Dict with the identifiers of the HLE questions asked as keys and a bool
    # representing if the question has been answered correctly.
    answers: "dict[str, dict[str, Union[str, bool]]]" = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class Result:
    """
    Dataclass for representing an entire run with potentially several models.
    """

    models: "list[str]" = dataclasses.field(default_factory=list)

    # One dict entry for each model tested
    model_results: "dict[str, IndividualResult]" = dataclasses.field(
        default_factory=dict
    )


def write_result_file(run_id: str, result: Result) -> None:
    """
    Writes a result object to a file.
    """

    filename: str = f"{run_id}-results.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": Result.models,
                "results": {
                    k: {
                        "correct": res.correct,
                        "wrong": res.wrong,
                        "total": res.correct + res.wrong,
                        "ratio": (res.correct / (res.correct + res.wrong)),
                        "answers": res.answers,
                    }
                    for k, res in result.model_results.items()
                },
            },
            f,
        )


def prompt_model(client, model, question) -> ollama.ChatResponse:
    """
    Prompts a given model with a question.
    """

    if question["image"]:  # "" if not multi-modal
        image = {"type": "image_url", "image_url": {"url": question["image"]}}

        content = [question["question"], image]
    else:
        content = [question["question"]]

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    SYSTEM_EXACT_ANSWER
                    if question["answer_type"] == "exactMatch"
                    else SYSTEM_MC
                ),
            },
            {"role": "user", "content": content},
        ],
    )

    return response


def prompt_judge_model(
    client, model, question, answer, actual_response
) -> bool:
    """
    Prompts a third party about whether the answer to a given question is
    correct or not.
    """

    while True:
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question,
                        correct_answer=answer,
                        response=actual_response,
                    ),
                },
            ],
        )

        text = response["message"]["content"]

        if "yes, correct" in text.lower():
            return True
        elif "no, incorrect" in text.lower():
            return False


def main():
    """
    The main function that manages the entire program flow.
    """

    args = p.parse_args()
    models: "list[str]" = [model.strip() for model in args.model.split(",")]

    client = ollama.Client(
        host=os.getenv("HLE_OLLAMA_HOST", "http://localhost:11434"),
        headers={
            "Authentication": f"Bearer \
{os.getenv('HLE_OLLAMA_TOKEN', 'ollama')}"
        },
    )

    logger = create_logger()

    logger.info("loading dataset")
    questions = datasets.load_dataset(  # type: ignore
        STD_DATASET if args.dataset is None else args.dataset, split="test"
    ).to_list()  # type: ignore

    if args.num_questions is not None:
        logger.info(
            "picking %s random questions for this run", args.num_questions
        )
        questions = random.choices(questions, k=args.num_questions)

    glob_res = Result(models=models)

    logger.info("prompting models for answers")
    for model in models:
        res = IndividualResult()
        logger.info("prompting %s with the questions", model)

        for question in tqdm.tqdm(questions):
            try:
                response = prompt_model(client, model, question)

                answer: str = response["message"]["content"]

                res.answers[question["id"]] = {
                    "answer": answer,
                    "correct": False,  # we do not know this yet!
                }
            except Exception as e:
                # Rare, but possible!
                logger.error("An error occured! %s", e)

        glob_res.model_results[model] = res

    logger.info("answer generation done; judging will now begin")

    for model in models:
        logger.info("judging %s's responses", model)
        model_results = glob_res.model_results[model]

        for question in tqdm.tqdm(questions):
            try:
                is_correct = prompt_judge_model(
                    client,
                    args.judge,
                    question["question"],
                    question["answer"],
                    model_results.answers[question["id"]]["answer"],
                )
            except KeyError as e:
                logger.error("KeyError: %s", e)
                continue

            model_results.answers[question["id"]]["correct"] = is_correct

    logger.info("done judging, finalizing results...")


if __name__ == "__main__":
    main()
