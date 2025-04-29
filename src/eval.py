"""
Evaluation script to run "Humanity's Last Exam" on Ollama models.
"""

import random
import time
from typing import Any, Union
import json
import dataclasses
import uuid
import argparse
import os
import ollama

import datasets
import tqdm
from logger import create_logger


p = argparse.ArgumentParser(
    "hle-eval-ollama",
    description="A simple-to-use evaluation program to get up and running \
with Humanity's Last Exam and Ollama models.",
)
p.add_argument(
    "--model",
    type=str,
    required=True,
    help="the model(s) to benchmark. Several models must be separated with \
commas.",
)
p.add_argument(
    "--judge",
    type=str,
    help="the model to be used for judging the other model's responses when \
given the correct answer. It is recommended not to use a model of a family \
already present in the --model argument.",
)
p.add_argument(
    "--dataset",
    type=str,
    help="override if you want to use something else than HLE (needs to be in \
the same format).",
)
p.add_argument(
    "--num-questions",
    type=int,
    help="how many questions shall be selected randomly (if argument is not \
given, all questions in the dataset are used). Please make sure not to \
execute separate runs with this option provided and then compare the results, \
as they would be non-representative due to their random nature.",
)
p.add_argument(
    "--only-text",
    action=argparse.BooleanOptionalAction,
    help="whether to use the text-only subset of the dataset; useful if the \
models provided are not multi-modal.",
)
p.add_argument(
    "--skip-judge",
    action=argparse.BooleanOptionalAction,
    help="whether to skip judging the responses (just writes them to a file).",
)
p.add_argument(
    "--add-question-info",
    action=argparse.BooleanOptionalAction,
    help="whether to add question and correct answer to the output file.",
)


logger = create_logger()


STD_DATASET: str = "cais/hle"

# Maximum number of tokens in an answer: ~8k
MAX_TOKENS_ANSWER: int = 2**13
# Maximum number of tokens when judging: ~4k
MAX_TOKENS_JUDGE: int = 2**12

ERROR_TIMEOUT: int = 3  # in seconds; wait period before next request is sent

# taken from line 11-13,
# https://github.com/centerforaisafety/hle
SYSTEM_EXACT_ANSWER = """Your response should be in the following format:\n\
Explanation: {your explanation for your final answer}\nExact Answer: {your \
succinct, final answer}"""

SYSTEM_MC = """Your response should be in the following format:\nExplanation: \
{your explanation for your answer choice}\nAnswer: {your chosen answer}\n"""


# taken partly from line 16-33,
# https://github.com/centerforaisafety/hle
JUDGE_PROMPT = """You're a teacher correcting a student's exam. You are given \
the [exam question], the [student's response] and the known, [correct answer] \
to compare it to.

[exam question]: {question}

=== === ===

[student's response]: {response}

=== === ===

[correct answer]: {correct_answer}

=== === ===

Please extract the exact, final answer from the [student's response]. Then,
given the [correct answer], judge whether that final answer is correct or \
not, focusing only on if there are meaningful differences between \
[correct answer] and the extracted student's final answer.

DO NOT comment on any background to the problem, DO NOT attempt to solve the \
problem, DO NOT argue for any answer different than [correct answer], focus \
only on whether the answers match.

Respond in valid JSON.
"""

JUDGE_FORMAT: "dict[str, Any]" = {
    "type": "object",
    "properties": {
        "extracted_final_answer": {"type": "string"},
        "is_answer_correct": {"type": "boolean"},
    },
    "required": ["extracted_final_answer", "is_answer_correct"],
}


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

    filename: str = f"{run_id}.results.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": result.models,
                "results": {
                    k: {
                        "correct": res.correct,
                        "wrong": res.wrong,
                        "total": res.correct + res.wrong,
                        "ratio": (
                            (res.correct / (res.correct + res.wrong))
                            if (res.correct + res.wrong != 0)
                            else 0
                        ),
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

    user_message = {"role": "user", "content": question["question"]}

    if question["image"]:
        # attach an image if there is one in the question
        user_message["images"] = [question["image"].split(",")[-1]]

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
            user_message,
        ],
        options={"num_predict": MAX_TOKENS_ANSWER},
    )

    return response


def prompt_judge_model(
    client, model, question, answer, actual_response
) -> bool:
    """
    Prompts a third party about whether the answer to a given question is
    correct or not.
    """

    if not answer.strip():
        # No answer at all - that's wrong.
        return False

    while True:
        # (as long as the judge model is unsure and hasn't given a definitive
        # answer yet)

        try:
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
                options={"num_predict": MAX_TOKENS_JUDGE},
                format=JUDGE_FORMAT,
            )

            return json.loads(response.message.content)["is_answer_correct"]
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error while judging: %s", e)
            time.sleep(ERROR_TIMEOUT)


def finalize_result(glob_res: Result):
    """
    Finalizes the results by counting wrong and correct answers, then writes
    those back into the Result object.
    """

    for res in glob_res.model_results.values():
        total_answers = len(res.answers)
        correct_answers = sum(
            (1 if el["correct"] else 0) for el in res.answers.values()
        )
        wrong_answers = total_answers - correct_answers

        res.correct = correct_answers
        res.wrong = wrong_answers


def judge_answers(
    args,
    models: "list[str]",
    client: ollama.Client,
    questions,
    glob_res: Result,
):
    """
    Makes a - hopefully independent, third-party - model judge the other
    models' responses.
    """

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


def generate_anwers(
    models: "list[str]",
    client: ollama.Client,
    questions,
    glob_res: Result,
    add_question_info: bool = False,
):
    """
    Prompts the models for answers to the questions in the dataset one by one.
    """

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

                if add_question_info:
                    res.answers[question["id"]].update(
                        {
                            "question_text": question["question"],
                            "correct_answer": question["answer"],
                        }
                    )
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Rare, but possible!
                logger.error("An error occured! %s", e)
                time.sleep(ERROR_TIMEOUT)

        glob_res.model_results[model] = res


def print_results(models: "list[str]", glob_res: Result):
    """
    Prints the results of the run.
    """

    for model in models:
        model_results = glob_res.model_results[model]
        logger.info(
            "%s: %s correct, %s wrong (%s percent)",
            model,
            model_results.correct,
            model_results.wrong,
            round(
                100
                * (
                    (
                        model_results.correct
                        / (model_results.correct + model_results.wrong)
                        if (model_results.correct + model_results.wrong) != 0
                        else 0
                    )
                ),
                2,
            ),
        )


def main():
    """
    The main function that manages the entire program flow.
    """

    args = p.parse_args()

    assert (not args.skip_judge) or (
        args.skip_judge and not args.judge
    ), "cannot specify a judge when --skip-judge is given."

    assert (
        (args.skip_judge) or (not args.skip_judge) and args.judge
    ), "judge model needs to be specified."

    run_uuid = str(uuid.uuid4())
    logger.info("run uuid is %s", run_uuid)

    models: "list[str]" = [model.strip() for model in args.model.split(",")]

    client = ollama.Client(
        host=os.getenv("HLE_OLLAMA_HOST", "http://localhost:11434"),
        headers={
            "Authentication": f"Bearer \
{os.getenv('HLE_OLLAMA_TOKEN', 'ollama')}"
        },
    )

    logger.info("loading dataset")
    questions = datasets.load_dataset(  # type: ignore
        STD_DATASET if args.dataset is None else args.dataset, split="test"
    ).to_list()  # type: ignore

    if args.only_text:
        questions = list(filter(lambda el: not el["image"], questions))

    if args.num_questions is not None:
        logger.info(
            "picking %s random questions for this run", args.num_questions
        )
        questions = random.choices(questions, k=args.num_questions)

    glob_res = Result(models=models)

    logger.info("prompting models for answers")
    generate_anwers(
        models, client, questions, glob_res, args.add_question_info
    )

    logger.info("temporarily saving results")
    write_result_file(run_uuid, glob_res)

    if args.skip_judge:
        logger.info("--skip-judge specified, leaving")
        return

    logger.info("judging model answers")
    judge_answers(args, models, client, questions, glob_res)

    logger.info("done judging, finalizing results")
    finalize_result(glob_res)

    logger.info("writing results")
    write_result_file(run_uuid, glob_res)

    print_results(models, glob_res)


if __name__ == "__main__":
    main()
