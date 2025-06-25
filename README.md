# llm-utils
Useful utilities for LLM based applications. 

## Development

This project uses `uv` for dependency management and as a runner.

### Setup

To set up the development environment, run the following commands:

```bash
uv sync
uv run python -m pip install -e .
```

This will install all the dependencies and make the `llmutils` package available to the test runner.

### Running Tests

To run the tests, use the following command:

```bash
uv run pytest
``` 

## LLM Healing Evaluation

The `scripts/evaluate_llm_healing.py` script is used to evaluate the performance of different LLMs and prompt strategies for healing broken text. The script can be run with different dataset types, models, and strategies.

### Usage

```bash
uv run python scripts/evaluate_llm_healing.py <dataset_type> [--models <models>] [--strategies <strategies>]
```

- `dataset_type`: The type of dataset to evaluate. Choices: `json`, `clue_answer`, `list_of_strings`.
- `--models`: A comma-separated list of models to test.
- `--strategies`: A comma-separated list of strategies to test.

### Latest Results (JSON Dataset)

| Model                                      | simple | with_instructions | with_examples |
| ------------------------------------------ | ------ | ----------------- | ------------- |
| openai/gpt-4.1-mini                        | 6/10   | 7/10              | 8/10          |
| meta-llama/llama-3-8b-instruct             | 2/10   | 7/10              | 8/10          |
| google/gemini-2.5-flash-lite-preview-06-17 | 3/10   | 7/10              | 8/10          |
| openai/gpt-4.1-nano                        | 8/10   | 8/10              | 8/10          |
| anthropic/claude-3-haiku                   | 5/10   | 8/10              | 8/10          |
| deepseek/deepseek-chat-v3-0324             | 2/10   | 8/10              | 9/10          |