<p align="center">
    <img src="./images/hle_logo.png" width=120>
</p>

# hle-eval-ollama

<p align="center">
    <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/mags0ft/hle-eval-ollama/pylint.yml?style=for-the-badge&logo=python&labelColor=%231e1e1e" />
</p>

This repo aims to allow anyone to get up and running with [Humanity's Last Exam](https://lastexam.ai/) and Ollama locally.

The official repo with evaluation scripts by HLE is notoriously hard to use, only lightly documented and merely made to work with the OpenAI API. Re-writing it to work with more AI inference providers and APIs would only add more complexity, which is why I chose to create this repository.

## How to use it

It's luckily simple! First of all, make sure to have a [Huggingface](https://huggingface.co/) account. The HLE dataset is **gated**, which means that you will need to authenticate in order to use it.

```bash
python3 -m venv .venv
. ./.venv/bin/activate
pip install -r ./requirements.txt
```

Generate a HuggingFace access token [here](https://huggingface.co/settings/tokens) and copy it to your clipboard.
Then, run

```
huggingface-cli login
```

Now you're all set! For example, run

```bash
python3 ./src/eval.py --model=gemma3 --num-questions=150
```

to begin the exam for the model! Results will also be written to an output file in the project root directory, ending in `-results.json`.

**Tip**: You can specify several models separated by commata in order to make them compete against each other.

**Important**: Do not just perform separate runs with `--num-questions` specified, as this will pick different, random questions from the dataset for each run individually. If you want to compare models with a limited number of questions, use the tip described above.

## Environment variables

- `HLE_OLLAMA_HOST`: specifies the Ollama host to connect to.
- `HLE_OLLAMA_TOKEN`: specifies the Bearer token to use for Ollama authentication.

## Thanks

Huge thanks to the [creators](https://github.com/centerforaisafety/hle/blob/main/citation.txt) of Humanity's last exam for the extraordinarily hard questions!

Also, huge thanks to the Ollama contributors and creators of all the [packages](./requirements.txt) used for this project!
