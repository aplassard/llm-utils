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
