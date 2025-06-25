import asyncio
import logging
import random
from typing import Any, Callable, Tuple, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variable for the return type of the LLM call
R = TypeVar("R")


class LLMError(Exception):
    """Custom exception for LLM errors."""
    pass


@retry(
    retry=retry_if_exception_type(LLMError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True,  # Crucial: ensures Tenacity re-raises the last caught exception directly
)
async def llm_call_with_retry(
    llm_call_func: Callable[..., R], *args: Tuple[Any, ...], **kwargs: Any
) -> R:
    """
    Calls an LLM function with retries on failure.

    Args:
        llm_call_func: The LLM function to call.
        *args: Positional arguments to pass to the LLM function.
        **kwargs: Keyword arguments to pass to the LLM function.
                  A special kwarg `_test_simulate_internal_error_random_check` (boolean)
                  can be passed to enable/disable the internal random error simulation
                  for testing purposes. Defaults to False if not provided.

    Returns:
        The result of the LLM function call.

    Raises:
        LLMError: If the LLM call (or the internal random simulation) fails
                  after the maximum number of retries. Tenacity, with reraise=True,
                  will re-throw the last LLMError it encountered.
    """
    # Pop the test-specific kwarg to avoid passing it to the actual llm_call_func
    simulate_internal_error_check = kwargs.pop('_test_simulate_internal_error_random_check', False)

    try:
        result = await llm_call_func(*args, **kwargs)

        if simulate_internal_error_check:
            if random.random() < 0.6:  # 60% chance of error
                logger.info("Simulating internal LLMError in llm_call_with_retry's random check.")
                raise LLMError("Simulated LLM error by llm_call_with_retry's own random check")
        return result
    except LLMError as e:
        # This catches LLMError from llm_call_func or the random check above.
        # Log and re-raise for Tenacity to handle.
        logger.warning(f"LLM call attempt failed with LLMError: {e}. Tenacity will retry if applicable.")
        raise
    except Exception as e:
        # This catches any other exception from llm_call_func.
        # Wrap it in LLMError so Tenacity's retry logic applies.
        logger.error(f"An unexpected error occurred during LLM call attempt: {e}. Wrapping in LLMError.")
        raise LLMError(f"Unexpected error during attempt: {e}") from e
