<p align="center">
    <img src="./images/showcase.webp" alt="Logo of Humanity's Last Exam on a white, circular background." width=480>
</p>

<h1 align="center">hle-eval-ollama</h1>

<p align="center">
    <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/mags0ft/hle-eval-ollama/pylint.yml?style=for-the-badge&logo=python&labelColor=%231e1e1e" />
</p>

_Want to see how your favorite local LLMs fair against Humanity's Last Exam? What difference quantizations may make?_

This repo aims to **allow anyone to get up and running with [Humanity's Last Exam](https://lastexam.ai/) (or similar benchmarks!) and Ollama locally**.

The official repo with evaluation scripts by HLE is notoriously hard to use, only lightly documented and merely made to work with the OpenAI API. While Ollama exposes an OpenAI API compatible endpoint, this project aims for a two-way approach, featuring both a pure, Ollama-agnostic API implementation and an OpenAI API compatible backend to show what's possible.

> [!IMPORTANT]  
> The whole quality of the benchmark results bases on how good the judge model does its job. If it judges poorly, good models might look worse and bad models better. Make sure to choose a strong model and verify results yourself.
>
> There are ongoing problems with the quality of the judge model's responses. Often, answers are still misjudged. Please exercise caution or manually review until cutting-edge models are able to correctly identify correct and wrong responses consistently.

## How to use it

It's luckily simple! First of all, make sure to have a [Hugging Face](https://huggingface.co/) account. The HLE dataset is **gated**, which means that you will need to authenticate in order to use it. You may want to visit the [HLE page](https://huggingface.co/datasets/cais/hle) on Hugging Face and agree to your information being submitted.

```bash
python3 -m venv .venv
. ./.venv/bin/activate
pip install -r ./requirements.txt
```

Generate a Hugging Face access token [here](https://huggingface.co/settings/tokens) and copy it to your clipboard.
Then, run

```
huggingface-cli login
```

Now you're all set! For example, run

```bash
python3 ./src/eval.py --model=gemma3 --judge=llama3:8b --num-questions=150
```

to begin the exam for the model! Results will also be written to an output file in the project root directory, ending in `.results.json`.

**Tip**: You can specify several models separated by commas in order to make them compete against each other. You can - and must - only specify one judge model (the model that will rate the answers), and it's highly recommended to choose a model that isn't part of the models taking the exam.

**Important**: Do not just perform separate runs with `--num-questions` specified, as this will pick different, random questions from the dataset for each run individually. If you want to compare models with a limited number of questions, use the tip described above.

**For text-only models** specify `--only-text` to only use the text subset of the HLE dataset.

> [!TIP]
> You can also use any OpenAI API compatible endpoint by providing the `--backend=openai` flag. Make sure to set the `HLE_EVAL_API_KEY` and `HLE_EVAL_ENDPOINT` environment variables.
>
> **PLEASE NOTE** that image input (vision) is still unstable for OpenAI endpoints - while it works, it consumes an absurd amount of tokens (which you may be billed for!) and is not recommended for use. You can use a lighter variant by setting `USE_EXPERIMENTAL_IMAGE_UPLOAD` to `True` in `src/constants.py`, but this does not work for every endpoint.

## Example results

Below is a comparison of two commonly used models; they've been asked 100 questions from the text-only subset and the answers were judged by Phi-4 (14b) by Microsoft.

**These are by no means professionally taken, high-quality results and thus not representative.**

```
$ python3 ./src/eval.py --model=llama3:8b,mistral:7b --judge=phi4:latest --num-questions=100 --only-text
[...]
hle-eval-ollama: INFO - llama3:8b: 7 correct, 89 wrong (7.29 percent)
hle-eval-ollama: INFO - mistral:7b: 13 correct, 83 wrong (13.54 percent)
```

<p align="center">
    <img src="./images/comparison.png" width=512 alt="Image comparing these results visually in a bar diagram.">
</p>

_Note: these benchmark results have been captured using automatic response judging, which - as mentioned above - is still relatively unreliable. The results are not representative and shall not be cited._

## Environment variables

- `HLE_EVAL_ENDPOINT`: specifies the host to connect to.
- `HLE_EVAL_API_KEY`: specifies the Bearer token to use for authentication.

## Thanks

Huge thanks to the [creators](https://github.com/centerforaisafety/hle/blob/main/citation.txt) of Humanity's last exam for the extraordinarily hard questions!

Also, huge thanks to the Ollama contributors and creators of all the [packages](./requirements.txt) used for this project!
