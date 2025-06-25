import pytest
from unittest.mock import patch
from llmutils.self_healing import heal_llm_output
from tenacity import RetryError

@patch('llmutils.self_healing.call_llm_with_retry')
def test_heal_llm_output_success(mock_call_llm_with_retry):
    """
    Tests the heal_llm_output function for a successful LLM call.
    """
    # Arrange
    broken_text = "This is sme text with typoes."
    prompt_template = "Fix the typos: {broken_text}"
    expected_healed_text = "This is some text with typos."
    mock_call_llm_with_retry.return_value = expected_healed_text

    # Act
    healed_text = heal_llm_output(broken_text, prompt_template, model_name="test/model")

    # Assert
    mock_call_llm_with_retry.assert_called_once_with(
        "test/model",
        prompt_template.format(broken_text=broken_text)
    )
    assert healed_text == expected_healed_text

@patch('llmutils.self_healing.call_llm_with_retry')
def test_heal_llm_output_llm_call_fails(mock_call_llm_with_retry):
    """
    Tests that heal_llm_output raises RetryError if the LLM call fails.
    """
    # Arrange
    broken_text = "This will fail."
    prompt_template = "Fix this: {broken_text}"
    mock_call_llm_with_retry.side_effect = RetryError("LLM API Error")

    # Act & Assert
    with pytest.raises(RetryError) as excinfo:
        heal_llm_output(broken_text, prompt_template, model_name="test/failing_model")

    assert "LLM API Error" in str(excinfo.value)
    mock_call_llm_with_retry.assert_called_once()

