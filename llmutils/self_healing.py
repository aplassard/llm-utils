import os
import logging
from llmutils.llm_with_retry import call_llm_with_retry

# Configure basic logging
logger = logging.getLogger(__name__)

def build_healing_prompt(
    broken_text: str,
    expected_format: str,
    instructions: str | None = None,
    good_examples: list[str] | None = None,
    bad_examples: list[str] | None = None,
    parsing_code: str | None = None,
    call_to_action: str = "Clean the following text:",
) -> str:
    """Builds the prompt for the LLM to heal the broken text."""
    prompt_parts = []

    if instructions:
        prompt_parts.append(f"<instructions>{instructions}</instructions>")
    else:
        prompt_parts.append("<instructions>Your task is to correct the provided text to match the specified format. Analyze the examples to understand the desired output.</instructions>")

    prompt_parts.append(f"<expected_format>{expected_format}</expected_format>")

    if good_examples:
        prompt_parts.append("These are examples of successful transformations. The text was corrected to match the expected format.")
        prompt_parts.append("<good_examples>")
        for example in good_examples:
            prompt_parts.append(f"<example>{example}</example>")
        prompt_parts.append("</good_examples>")

    if bad_examples:
        prompt_parts.append("These are examples of unsuccessful transformations. The original text could not be corrected because the necessary information was missing.")
        prompt_parts.append("<bad_examples>")
        for example in bad_examples:
            prompt_parts.append(f"<example>{example}</example>")
        prompt_parts.append("</bad_examples>")

    if parsing_code:
        prompt_parts.append("The following Python code is used to parse the text. The corrected text should be parsable by this code.")
        prompt_parts.append(f"<parsing_code>```python\n{parsing_code}\n```</parsing_code>")

    prompt_parts.append(f"{call_to_action}")
    prompt_parts.append(f"<broken_text>{broken_text}</broken_text>")

    return "\n".join(prompt_parts)

def heal_llm_output(
    broken_text: str,
    expected_format: str,
    instructions: str | None = None,
    good_examples: list[str] | None = None,
    bad_examples: list[str] | None = None,
    parsing_code: str | None = None,
    call_to_action: str = "Clean the following text:",
    model_name: str = "openai/gpt-4.1-nano",
) -> str:
    """
    Takes malformed text and uses an LLM to correct its structure based on a provided prompt template.

    Args:
        broken_text: The text that needs to be healed.
        expected_format: A description of the data format that is expected to be returned.
        instructions: A set of instructions that describe that the goal of the llm is to clean the data.
        good_examples: One or more examples of text that should be cleaned.
        bad_examples: One or more examples where we would not expect it to clean successfully.
        parsing_code: A string of text that is the code used to parse the returned text.
        call_to_action: The call to action for the LLM.
        model_name: The name of the LLM model to use (default: "openai/gpt-4.1-nano").

    Returns:
        The healed text.

    Raises:
        tenacity.RetryError: If the LLM call fails after all retries.
    """
    prompt_message = build_healing_prompt(
        broken_text=broken_text,
        expected_format=expected_format,
        instructions=instructions,
        good_examples=good_examples,
        bad_examples=bad_examples,
        parsing_code=parsing_code,
        call_to_action=call_to_action,
    )

    logger.info(f"Attempting to heal LLM output with model: {model_name}...")
    healed_text = call_llm_with_retry(model_name, prompt_message)
    logger.info("LLM healing call successful.")
    return healed_text