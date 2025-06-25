import os
import logging
from llmutils.llm_with_retry import call_llm_with_retry

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def heal_llm_output(broken_text: str, prompt_template: str, model_name: str = "openai/gpt-4.1-nano") -> str:
    """
    Takes malformed text and uses an LLM to correct its structure based on a provided prompt template.

    Args:
        broken_text: The text that needs to be healed.
        prompt_template: A string template for the prompt. It should be formattable with `broken_text`.
                         Example: "Please fix this text: {broken_text}"
        model_name: The name of the LLM model to use (default: "openai/gpt-4.1-nano").

    Returns:
        The healed text.

    Raises:
        tenacity.RetryError: If the LLM call fails after all retries.
    """
    prompt_message = prompt_template.format(broken_text=broken_text)

    logger.info(f"Attempting to heal LLM output with model: {model_name}...")
    healed_text = call_llm_with_retry(model_name, prompt_message)
    logger.info("LLM healing call successful.")
    return healed_text

