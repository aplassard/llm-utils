# This file will contain the self_healing_text function.
import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

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
        Exception: If the LLM call fails.
    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY environment variable not set.")
        raise ValueError("OPENROUTER_API_KEY must be set in the environment.")

    prompt_message = prompt_template.format(broken_text=broken_text)

    logger.info(f"Attempting to heal LLM output with model: {model_name}...")
    try:
        llm = ChatOpenAI(
            model_name=model_name,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.environ.get("OPENROUTER_API_KEY")
        )
        response = llm.invoke([HumanMessage(content=prompt_message)])
        healed_text = response.content
        logger.info("LLM healing call successful.")
        return healed_text
    except Exception as e:
        logger.error(f"LLM healing call failed. Error: {e}")
        # Re-raising the exception to make it visible if healing fails.
        raise
