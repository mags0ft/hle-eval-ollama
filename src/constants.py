from typing import Any


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