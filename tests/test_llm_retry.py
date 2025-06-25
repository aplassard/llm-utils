import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from llmutils.llm_retry import llm_call_with_retry, LLMError


# A simple async function to simulate an LLM call
async def mock_llm_call(succeed: bool = True):
    await asyncio.sleep(0.01)  # Simulate some network latency
    if succeed:
        return "LLM call successful"
    else:
        raise LLMError("Simulated LLM error for testing")


@pytest.mark.asyncio
async def test_llm_call_succeeds_on_first_attempt():
    """Test that the function returns the expected output when the LLM call succeeds on the first attempt."""
    # Disable internal random check by passing _test_simulate_internal_error_random_check=False (default)
    # or by patching random.random if it were still global.
    # Since it's now controlled by a kwarg, no patch for random.random is needed here for the internal check.
    result = await llm_call_with_retry(mock_llm_call, succeed=True)
    assert result == "LLM call successful"


@pytest.mark.asyncio
async def test_llm_call_retries_and_succeeds():
    """Test that the function retries the LLM call and eventually succeeds."""
    mock_call = AsyncMock(
        side_effect=[
            LLMError("Simulated LLM error attempt 1"),
            LLMError("Simulated LLM error attempt 2"),
            "LLM call successful after retries",
        ]
    )
    # _test_simulate_internal_error_random_check defaults to False, so no random internal errors.
    result = await llm_call_with_retry(mock_call)
    assert result == "LLM call successful after retries"
    assert mock_call.call_count == 3


@pytest.mark.asyncio
async def test_llm_call_fails_after_max_retries():
    """Test that the function raises LLMError when the LLM call fails after the maximum number of retries."""
    mock_call = AsyncMock(side_effect=LLMError("Simulated LLM error"))

    # _test_simulate_internal_error_random_check defaults to False.
    with pytest.raises(LLMError) as excinfo:
        await llm_call_with_retry(mock_call)

    # Because reraise=True, the actual last LLMError is raised.
    assert "Simulated LLM error" in str(excinfo.value)
    # The llm_call_with_retry is configured with stop_after_attempt(5)
    assert mock_call.call_count == 5


@pytest.mark.asyncio
async def test_llm_call_with_internal_simulated_failure_and_retry():
    """
    Test the internal random failure simulation within llm_call_with_retry.
    This test expects the call to fail a few times due to the random check
    and then succeed.
    """
    successful_call_result = "Success after internal retries"
    async def llm_func_always_succeeds_externally():
        await asyncio.sleep(0.01)
        return successful_call_result

    # We'll mock random.random specifically for the internal check in llm_call_with_retry
    # Sequence: fail, fail, succeed (i.e., random.random() < 0.6 is True twice, then False)
    with patch('llmutils.llm_retry.random.random', side_effect=[0.1, 0.2, 0.9]) as mock_random:
         result = await llm_call_with_retry(
             llm_func_always_succeeds_externally,
             _test_simulate_internal_error_random_check=True
         )
    assert result == successful_call_result
    # llm_func_always_succeeds_externally is called 3 times (2 fails from random, 1 success)
    # random.random is called 3 times by the retry wrapper
    assert mock_random.call_count == 3


@pytest.mark.asyncio
async def test_llm_call_with_unexpected_error_is_wrapped_and_retried():
    """Test that an unexpected non-LLMError from the wrapped function is wrapped in LLMError and retried."""
    mock_call = AsyncMock(
        side_effect=[
            ValueError("Unexpected value error"), # Will be wrapped in LLMError
            "Successful after unexpected error",
        ]
    )
    # _test_simulate_internal_error_random_check defaults to False.
    result = await llm_call_with_retry(mock_call)
    assert result == "Successful after unexpected error"
    assert mock_call.call_count == 2

@pytest.mark.asyncio
async def test_llm_call_fails_permanently_with_unexpected_error():
    """Test that a persistent unexpected non-LLMError causes LLMError to be raised after retries."""
    mock_call = AsyncMock(side_effect=ValueError("Persistent unexpected value error"))

    # _test_simulate_internal_error_random_check defaults to False.
    with pytest.raises(LLMError) as excinfo:
        await llm_call_with_retry(mock_call)

    # The ValueError will be wrapped. Tenacity will retry based on this wrapped LLMError.
    # After exhaustion, Tenacity (with reraise=True) will raise the last caught error,
    # which is the LLMError wrapping the ValueError.
    assert "Unexpected error during attempt: Persistent unexpected value error" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError) # Check that the original error is chained.
    assert mock_call.call_count == 5 # Max attempts
