import pytest
from unittest.mock import patch
from llmutils.self_healing import heal_llm_output, build_healing_prompt
from tenacity import RetryError


def test_build_healing_prompt():
    """Tests the build_healing_prompt function."""
    prompt = build_healing_prompt(
        broken_text="some broken text",
        expected_format="a description of the format",
        instructions="some instructions",
        good_examples=["good example 1", "good example 2"],
        bad_examples=["bad example 1"],
        parsing_code="some parsing code",
        call_to_action="a call to action",
    )
    assert "<instructions>some instructions</instructions>" in prompt
    assert "<expected_format>a description of the format</expected_format>" in prompt
    assert "These are examples of successful transformations." in prompt
    assert "<good_examples>" in prompt
    assert "<example>good example 1</example>" in prompt
    assert "<example>good example 2</example>" in prompt
    assert "</good_examples>" in prompt
    assert "These are examples of unsuccessful transformations." in prompt
    assert "<bad_examples>" in prompt
    assert "<example>bad example 1</example>" in prompt
    assert "</bad_examples>" in prompt
    assert "The following Python code is used to parse the text." in prompt
    assert "<parsing_code>```python\nsome parsing code\n```</parsing_code>" in prompt
    assert "a call to action" in prompt
    assert "<broken_text>some broken text</broken_text>" in prompt
    assert "</prompt>" not in prompt
    assert "</call_to_action>" not in prompt


def test_build_healing_prompt_defaults():
    """Tests the build_healing_prompt function with default values."""
    prompt = build_healing_prompt(
        broken_text="some broken text",
        expected_format="a description of the format",
    )
    assert "<instructions>Your task is to correct the provided text to match the specified format. Analyze the examples to understand the desired output.</instructions>" in prompt
    assert "<expected_format>a description of the format</expected_format>" in prompt
    assert "<good_examples>" not in prompt
    assert "<bad_examples>" not in prompt
    assert "<parsing_code>" not in prompt
    assert "Clean the following text:" in prompt
    assert "<broken_text>some broken text</broken_text>" in prompt


@patch('llmutils.self_healing.call_llm_with_retry')
def test_heal_llm_output_success(mock_call_llm_with_retry):
    """
    Tests the heal_llm_output function for a successful LLM call.
    """
    # Arrange
    broken_text = "This is sme text with typoes."
    expected_healed_text = "This is some text with typos."
    mock_call_llm_with_retry.return_value = expected_healed_text

    # Act
    healed_text = heal_llm_output(
        broken_text=broken_text,
        expected_format="No typos",
        model_name="test/model",
    )

    # Assert
    assert healed_text == expected_healed_text
    mock_call_llm_with_retry.assert_called_once()


@patch('llmutils.self_healing.call_llm_with_retry')
def test_heal_llm_output_llm_call_fails(mock_call_llm_with_retry):
    """
    Tests that heal_llm_output raises RetryError if the LLM call fails.
    """
    # Arrange
    broken_text = "This will fail."
    mock_call_llm_with_retry.side_effect = RetryError("LLM API Error")

    # Act & Assert
    with pytest.raises(RetryError) as excinfo:
        heal_llm_output(
            broken_text=broken_text,
            expected_format="This should fail",
            model_name="test/failing_model",
        )

    assert "LLM API Error" in str(excinfo.value)
    mock_call_llm_with_retry.assert_called_once()
