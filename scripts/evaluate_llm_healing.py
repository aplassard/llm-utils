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
PROMPT_FILE = "./scripts/two-shot-prompt.md"

MODELS_TO_TEST = [                                                                                 
"openai/gpt-4.1-mini",
"meta-llama/llama-3-8b-instruct",
"google/gemini-2.5-flash-lite-preview-06-17",
"openai/gpt-4.1-nano",
"anthropic/claude-3-haiku",
"deepseek/deepseek-chat-v3-0324"

]

def load_prompts(dataset_type):
    """Loads all prompts to be tested."""
    if dataset_type == 'json':
        prompts = {
            "simple": "Fix the following text to be valid JSON: {broken_text}",
            "explicit": "The following text is supposed to be a single, valid JSON object, but it is malformed. Correct the syntax errors. Only output the corrected JSON object, with no other text or explanation.\n\n{broken_text}",
            "role_playing": "You are an expert system that corrects malformed JSON. Your only task is to take the input text and output a syntactically correct JSON object. Do not add any commentary. Here is the input:\n\n{broken_text}"
        }
        with open(PROMPT_FILE, 'r') as f:
            prompts['two_shot'] = f.read()
    elif dataset_type == 'clue_answer':
        prompts = {
            "simple": "Extract the clue and answer from the following text and format it as 'clue: <clue_id>\nanswer: <answer_id>': {broken_text}",
            "explicit": "The following text contains a clue and an answer. Your task is to extract them and format them exactly as 'clue: <clue_id>\nanswer: <answer_id>'. Do not include any other text or explanation.\n\n{broken_text}"
        }
    elif dataset_type == 'list_of_strings':
        prompts = {
            "simple": "Extract the list of items from the following text and format them as a newline-separated list: {broken_text}",
            "explicit": "The following text contains a list of items. Your task is to extract them and format them as a newline-separated list, with no other text or explanation.\n\n{broken_text}"
        }
    return prompts

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
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

def is_valid_clue_answer(text):
    """Checks if a string is a valid clue/answer format."""
    pattern = re.compile(r"^clue: .*\\nanswer: .*$", re.IGNORECASE)
    return bool(pattern.match(text.strip()))

def is_valid_list(text):
    """Checks if a string is a valid newline-separated list."""
    # A valid list has at least one non-empty line and no special list markers (like - or *)
    lines = text.strip().split('\n')
    if not lines or not all(lines):
        return False
    return all(not line.strip().startswith(('-', '*', '.')) for line in lines)

def run_evaluation(dataset_type):
    """Runs the evaluation and prints a report."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY environment variable not set. Cannot run evaluation.")
        return

    data_dir = os.path.join(DATA_DIR_BASE, dataset_type)
    dataset = load_dataset(data_dir)
    prompts_to_test = load_prompts(dataset_type)
    results = {}

    if dataset_type == 'json':
        validation_fn = is_valid_json
    elif dataset_type == 'clue_answer':
        validation_fn = is_valid_clue_answer
    else:
        validation_fn = is_valid_list

    for model_name in MODELS_TO_TEST:
        results[model_name] = {}
        for prompt_name, prompt_template in prompts_to_test.items():
            logger.info(f"--- Testing Model: {model_name}, Prompt: {prompt_name} ---")
            success_count = 0
            total_count = 0

            for item in dataset:
                total_count += 1
                try:
                    healed_text = heal_llm_output(item['broken'], prompt_template, model_name)
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

            results[model_name][prompt_name] = {
                "success_rate": f"{success_count}/{total_count}"
            }

    # Print final report
    print("\n--- Evaluation Report ---")
    for model_name, prompts in results.items():
        print(f"\nModel: {model_name}")
        for prompt_name, result in prompts.items():
            print(f"  Prompt: {prompt_name:<15} | Success Rate: {result['success_rate']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM self-healing capabilities.")
    parser.add_argument("dataset_type", choices=["json", "clue_answer", "list_of_strings"], help="The type of dataset to evaluate.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_evaluation(args.dataset_type)


