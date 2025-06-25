import os
import json
import logging
import argparse
import re
from llmutils.self_healing import heal_llm_output

# Configure logging
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR_BASE = "./data/llm_healing"

ALL_MODELS = [
    "openai/gpt-4.1-mini",
    "meta-llama/llama-3-8b-instruct",
    "google/gemini-2.5-flash-lite-preview-06-17",
    "openai/gpt-4.1-nano",
    "anthropic/claude-3-haiku",
    "deepseek/deepseek-chat-v3-0324"
]

def get_prompt_strategies(dataset_type):
    """Loads all prompt strategies to be tested."""
    if dataset_type == 'json':
        strategies = {
            "simple": {
                "expected_format": "A single, valid JSON object.",
            },
            "with_instructions": {
                "expected_format": "A single, valid JSON object.",
                "instructions": "The following text is supposed to be a single, valid JSON object, but it is malformed. Correct the syntax errors. Only output the corrected JSON object, with no other text or explanation.",
            },
            "with_examples": {
                "expected_format": "A single, valid JSON object.",
                "instructions": "The following text is supposed to be a single, valid JSON object, but it is malformed. Correct the syntax errors. Only output the corrected JSON object, with no other text or explanation.",
                "good_examples": [
                    json.dumps({"broken": '{"key": "value",}', "healed": '{"key": "value"}'}),
                    json.dumps({"broken": "{'key': 'value'}", "healed": '{"key": "value"}'})
                ],
                "bad_examples": [
                    json.dumps({"broken": "This is not json", "healed": "INVALID_JSON"})
                ],
                "parsing_code": "import json\njson.loads(text)"
            }
        }
    elif dataset_type == 'clue_answer':
        strategies = {
            "simple": {
                "expected_format": "clue: <clue_id>\nanswer: <answer_id>",
            },
            "with_instructions": {
                "expected_format": "clue: <clue_id>\nanswer: <answer_id>",
                "instructions": "The following text contains a clue and an answer. Your task is to extract them and format them exactly as 'clue: <clue_id>\nanswer: <answer_id>'. Do not include any other text or explanation.",
            },
            "with_examples": {
                "expected_format": "clue: <clue_id>\nanswer: <answer_id>",
                "instructions": "The following text contains a clue and an answer. Your task is to extract them and format them exactly as 'clue: <clue_id>\nanswer: <answer_id>'. Do not include any other text or explanation.",
                "good_examples": [
                    json.dumps({"broken": "The clue is C1 and the answer is A1", "healed": "clue: C1\nanswer: A1"}),
                ],
                "bad_examples": [
                    json.dumps({"broken": "There is no answer here.", "healed": "clue: <not_found>\nanswer: <not_found>"}),
                ],
                "parsing_code": "import re\nre.match(r'^clue: .*\\nanswer: .*$', text.strip(), re.IGNORECASE)"
            }
        }
    elif dataset_type == 'list_of_strings':
        strategies = {
            "simple": {
                "expected_format": "A newline-separated list of strings.",
            },
            "with_instructions": {
                "expected_format": "A newline-separated list of strings.",
                "instructions": "The following text contains a list of items. Your task is to extract them and format them as a newline-separated list, with no other text or explanation.",
            },
            "with_examples": {
                "expected_format": "A newline-separated list of strings.",
                "instructions": "The following text contains a list of items. Your task is to extract them and format them as a newline-separated list, with no other text or explanation.",
                "good_examples": [
                    json.dumps({"broken": "- item 1\n- item 2", "healed": "item 1\nitem 2"}),
                ],
                "bad_examples": [
                    json.dumps({"broken": "There are no items here.", "healed": ""}),
                ]
            }
        }
    return strategies

def setup_logging(log_level):
    """Sets up the logging configuration."""
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(data_dir):
    """Loads the JSON dataset from the specified directory."""
    dataset = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), 'r') as f:
                data = json.load(f)
                data['filename'] = filename
                dataset.append(data)
    return dataset

def is_valid_json(text):
    """Checks if a string is valid JSON."""
    try:
        # Attempt to strip any markdown and then load
        match = re.search(r'```json\n(.*?)```', text, re.DOTALL)
        if match:
            text = match.group(1)
        json.loads(text)
        return True
    except (json.JSONDecodeError, AttributeError):
        return False

def is_valid_clue_answer(text):
    """Checks if a string is a valid clue/answer format."""
    pattern = re.compile(r"^clue: .*\nanswer: .*$", re.IGNORECASE | re.DOTALL)
    return bool(pattern.match(text.strip()))

def is_valid_list(text):
    """Checks if a string is a valid newline-separated list."""
    # A valid list has at least one non-empty line and no special list markers (like - or *)
    lines = text.strip().split('\n')
    if not lines or (len(lines) == 1 and not lines[0]):
        return True # Empty list is valid for not_healable cases
    if not all(lines):
        return False
    return all(not line.strip().startswith(('-', '*', '.')) for line in lines)

def run_evaluation(dataset_type, models_to_test, strategies_to_test):
    """Runs the evaluation and prints a report."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY environment variable not set. Cannot run evaluation.")
        return

    data_dir = os.path.join(DATA_DIR_BASE, dataset_type)
    dataset = load_dataset(data_dir)
    all_strategies = get_prompt_strategies(dataset_type)
    results = {}

    if dataset_type == 'json':
        validation_fn = is_valid_json
    elif dataset_type == 'clue_answer':
        validation_fn = is_valid_clue_answer
    else:
        validation_fn = is_valid_list

    if not strategies_to_test:
        strategies_to_test = all_strategies.keys()

    for model_name in models_to_test:
        results[model_name] = {}
        for strategy_name in strategies_to_test:
            if strategy_name not in all_strategies:
                logger.warning(f"Strategy '{strategy_name}' not found for dataset '{dataset_type}'. Skipping.")
                continue
            strategy_args = all_strategies[strategy_name]
            logger.info(f"--- Testing Model: {model_name}, Strategy: {strategy_name} ---")
            success_count = 0
            total_count = 0

            for item in dataset:
                total_count += 1
                try:
                    healed_text = heal_llm_output(
                        broken_text=item['broken'],
                        model_name=model_name,
                        **strategy_args
                    )
                    logger.debug(f"    LLM Output for {item['filename']}:\n---\n{healed_text}\n---")
                    is_healed = validation_fn(healed_text)

                    if item['expected'] == 'healable' and is_healed:
                        success_count += 1
                        logger.info(f"  [SUCCESS] {item['filename']}: Healed successfully.")
                    elif item['expected'] == 'not_healable' and not is_healed:
                        if dataset_type == 'json' and healed_text.strip() == "INVALID_JSON":
                            success_count += 1
                            logger.info(f"  [SUCCESS] {item['filename']}: Correctly identified as INVALID_JSON.")
                        else:
                            success_count += 1
                            logger.info(f"  [SUCCESS] {item['filename']}: Correctly identified as not healable.")
                    else:
                        logger.warning(f"  [FAILURE] {item['filename']}: Expected {item['expected']}, but result was {'healed' if is_healed else 'not_healed'}.")

                except Exception as e:
                    logger.error(f"  [ERROR] {item['filename']}: An error occurred: {e}")

            results[model_name][strategy_name] = {
                "success_rate": f"{success_count}/{total_count}"
            }

    # Print final report
    print("\n--- Evaluation Report ---")
    for model_name, strategies in results.items():
        print(f"\nModel: {model_name}")
        for strategy_name, result in strategies.items():
            print(f"  Strategy: {strategy_name:<20} | Success Rate: {result['success_rate']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM self-healing capabilities.")
    parser.add_argument("dataset_type", choices=["json", "clue_answer", "list_of_strings"], help="The type of dataset to evaluate.")
    parser.add_argument("--models", type=str, default=",".join(ALL_MODELS), help="A comma-separated list of models to test.")
    parser.add_argument("--strategies", type=str, default="", help=f"A comma-separated list of strategies to test. Available strategies: {list(get_prompt_strategies('json').keys())}")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    args = parser.parse_args()
    
    models_to_test = [model.strip() for model in args.models.split(',')]
    strategies_to_test = [strategy.strip() for strategy in args.strategies.split(',')] if args.strategies else []

    setup_logging(args.log_level)
    run_evaluation(args.dataset_type, models_to_test, strategies_to_test)