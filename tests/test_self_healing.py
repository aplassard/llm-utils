import pytest
import os
from unittest.mock import patch, MagicMock
from llmutils.self_healing import heal_llm_output
from langchain_core.messages import AIMessage

# Set a dummy API key for testing purposes if not already set
if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = "dummy_key_for_testing"

def test_heal_llm_output_success():
    """
    Tests the heal_llm_output function for a successful LLM call.
    """
    broken_text = "This is sme text with typoes."
    prompt_template = "Fix the typos: {broken_text}"
    expected_healed_text = "This is some text with typos."

    # Mock the ChatOpenAI class and its invoke method
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = AIMessage(content=expected_healed_text)

    with patch('llmutils.self_healing.ChatOpenAI', return_value=mock_llm_instance) as mock_chat_openai:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"}): # Ensure API key is set for the test
            healed_text = heal_llm_output(broken_text, prompt_template, model_name="test/model")

        # Assert that ChatOpenAI was called with the correct parameters
        mock_chat_openai.assert_called_once_with(
            model_name="test/model",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key="test_key"
        )

        # Assert that the invoke method was called with the correct prompt
        # The prompt is a list of HumanMessage objects
        args, _ = mock_llm_instance.invoke.call_args
        called_prompt_content = args[0][0].content
        expected_prompt_content = prompt_template.format(broken_text=broken_text)
        assert called_prompt_content == expected_prompt_content

        # Assert that the function returns the expected healed text
        assert healed_text == expected_healed_text

def test_heal_llm_output_api_key_missing():
    """
    Tests that heal_llm_output raises ValueError if OPENROUTER_API_KEY is not set.
    """
    broken_text = "Some text"
    prompt_template = "Fix: {broken_text}"

    with patch.dict(os.environ, {}, clear=True): # Clear environment variables for this test
        with pytest.raises(ValueError) as excinfo:
            heal_llm_output(broken_text, prompt_template)
        assert "OPENROUTER_API_KEY must be set" in str(excinfo.value)

def test_heal_llm_output_llm_call_fails():
    """
    Tests the heal_llm_output function when the LLM call raises an exception.
    """
    broken_text = "This will fail."
    prompt_template = "Fix this: {broken_text}"

    # Mock the ChatOpenAI class and its invoke method to raise an exception
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("LLM API Error")

    with patch('llmutils.self_healing.ChatOpenAI', return_value=mock_llm_instance) as mock_chat_openai:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"}):
            with pytest.raises(Exception) as excinfo:
                heal_llm_output(broken_text, prompt_template, model_name="test/failing_model")

            assert "LLM API Error" in str(excinfo.value)

        # Assert that ChatOpenAI was called
        mock_chat_openai.assert_called_once_with(
            model_name="test/failing_model",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key="test_key"
        )
        # Assert that invoke was called
        mock_llm_instance.invoke.assert_called_once()
