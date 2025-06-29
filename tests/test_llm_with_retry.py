import pytest
import os
from unittest.mock import patch, MagicMock
from llmutils.llm_with_retry import call_llm_with_retry
from langchain_core.messages import AIMessage
from tenacity import RetryError

# A custom marker for integration tests that make real API calls
# You can skip these tests by running: pytest -m "not integration"
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests as integration tests")

# Fixture to set a mock environment variable for patched tests
@pytest.fixture
def mock_api_key(monkeypatch):
    """Sets a mock API key for the duration of a test."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-api-key-for-testing")

@patch('llmutils.llm_with_retry.ChatOpenAI')
def test_call_llm_success_first_try(mock_chat_openai, mock_api_key):
    """
    Tests that the function returns correct content on a successful first API call.
    """
    # Arrange
    mock_response = AIMessage(content="This is a test response.")
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = mock_response
    mock_chat_openai.return_value = mock_llm_instance

    # Act
    result = call_llm_with_retry("test-model", "Hello, world!")

    # Assert
    assert result == "This is a test response."
    mock_chat_openai.assert_called_once_with(
        model_name="test-model",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key="mock-api-key-for-testing",
        model_kwargs={}
    )
    mock_llm_instance.invoke.assert_called_once()

@patch('llmutils.llm_with_retry.ChatOpenAI')
def test_call_llm_with_model_kwargs(mock_chat_openai, mock_api_key):
    """
    Tests that the function correctly passes model_kwargs to the ChatOpenAI constructor.
    """
    # Arrange
    mock_response = AIMessage(content="This is a test response with model_kwargs.")
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = mock_response
    mock_chat_openai.return_value = mock_llm_instance
    model_kwargs = {"temperature": 0.7, "top_p": 0.9}

    # Act
    result = call_llm_with_retry("test-model-kwargs", "Hello with kwargs!", model_kwargs=model_kwargs)

    # Assert
    assert result == "This is a test response with model_kwargs."
    mock_chat_openai.assert_called_once_with(
        model_name="test-model-kwargs",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key="mock-api-key-for-testing",
        model_kwargs=model_kwargs
    )
    mock_llm_instance.invoke.assert_called_once()

@patch('llmutils.llm_with_retry.ChatOpenAI')
def test_call_llm_success_after_retries(mock_chat_openai, mock_api_key):
    """
    Tests that the function succeeds after a few failed attempts.
    """
    # Arrange
    mock_response = AIMessage(content="Success after retries.")
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = [
        Exception("API Error 1"),
        Exception("API Error 2"),
        mock_response
    ]
    mock_chat_openai.return_value = mock_llm_instance

    # Act
    result = call_llm_with_retry("retry-model", "Please work this time.")

    # Assert
    assert result == "Success after retries."
    assert mock_llm_instance.invoke.call_count == 3

@patch('llmutils.llm_with_retry.ChatOpenAI')
def test_call_llm_fails_after_all_attempts(mock_chat_openai, mock_api_key):
    """
    Tests that the function raises a RetryError after all attempts fail.
    """
    # Arrange
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("Persistent API Error")
    mock_chat_openai.return_value = mock_llm_instance

    # Act & Assert
    with pytest.raises(RetryError) as excinfo:
        call_llm_with_retry("fail-model", "This will fail.")

    assert mock_llm_instance.invoke.call_count == 3
    # Check the cause of the RetryError for the original exception message
    assert "Persistent API Error" in str(excinfo.value.__cause__)

@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY environment variable not set for integration test"
)
def test_call_llm_with_retry_integration():
    """
    Tests the function by making a real API call to a fast and free model.
    This is an integration test and requires a valid OPENROUTER_API_KEY.
    """
    # Arrange
    # Use a model that is known to be fast and free on OpenRouter
    model_name = "mistralai/mistral-7b-instruct:free"
    prompt = "Hello! Respond with just one word: 'test'."

    # Act
    result = call_llm_with_retry(model_name, prompt)

    # Assert
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # A simple check to see if the model responded as expected
    assert "test" in result.lower()