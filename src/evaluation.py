import os
import json
import argparse
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from dataset import GSM8kDatasetLoader, MATHDatasetLoader
from training_utils import format_zero_shot_prompt, format_few_shot_prompt, generate_responses
from math_parser import compare_answers
from utils import convert_to_json
from model import load_model, load_model_with_flash_attention
from prompt import LLAMA_CHAT_TEMPLATE


def evaluate(args) -> None:
    """
    Main function to run the evaluation script.

    Parameters:
        args (Namespace): Configuration arguments
    """
    # Handle dataset-specifc loader
    if args.dataset == 'gsm8k':
        dataset_loader = GSM8kDatasetLoader()
    elif args.dataset == 'math':
        dataset_loader = MATHDatasetLoader(model_name=args.model_name)
    else:
        raise ValueError(f"Unsupported dataset: '{args.dataset}'. Please specify a valid dataset.")
    
    # Load and format dataset
    datasets = dataset_loader.load_from_json()
    datasets = datasets['test']
    
    # Format according to type
    if args.prompt != "direct":
        if args.prompt == "zero-shot":
            datasets = datasets.map(lambda x: format_zero_shot_prompt(x, args.prompt_system, args.model_name))
        elif args.prompt == "few-shot":
            datasets = datasets.map(lambda x: format_few_shot_prompt(x, args.few_shot_path))
        else:
            raise ValueError(f"Unsupported prompt: '{args.prompt}. Please specify a valid prompt type.")

    # Create output directory
    output_dir = f"data/{args.dataset}/{args.model_name}/results"
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenier
    accelerator = None
    if args.accelerate:
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        accelerator = Accelerator(kwargs_handlers=[kwargs])
        if args.flash_attention:
            model, tokenizer = load_model_with_flash_attention(args.model_path, {"": accelerator.process_index})
        else:
            model, tokenizer = load_model(args.model_path, {"": accelerator.process_index})
    else:
        if args.flash_attention:
            model, tokenizer = load_model_with_flash_attention(args.model_path, "auto")
        else:
            model, tokenizer = load_model(args.model_path, "auto")
        
    # Format prompts based on whether using direct input or chat template
    def format_prompt(example):
        if args.prompt == "direct":
            if "qwen" in args.model_name.lower():
                formatted_prompt = example['input']
            else:
                formatted_prompt = tokenizer.bos_token + example['input']
        else:
            if "llama" in args.model_name.lower():
                # Use chat template for other prompt types
                formatted_prompt = tokenizer.apply_chat_template(
                    example['messages'],
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=LLAMA_CHAT_TEMPLATE
                )
            else:
                formatted_prompt = tokenizer.apply_chat_template(
                    example['messages'],
                    tokenize=False,
                    add_generation_prompt=True
                )
        return {"prompt": formatted_prompt}
    
    datasets = datasets.map(format_prompt)
    if 'messages' in datasets.column_names:
        datasets = datasets.remove_columns(['messages'])
    
    # Greedy decoding
    args.temperature = None
    args.top_p = None
    args.top_k = None
    args.do_sample = False

    outputs, output_token_counts = generate_responses(datasets, model, tokenizer, args, accelerator)

    # Save to JSON
    if not args.accelerate or (args.accelerate and accelerator.is_main_process):
        file_path = convert_to_json(datasets, outputs, output_token_counts, output_dir, model_name=args.model_name)
    
    if not args.accelerate or (args.accelerate and accelerator.is_main_process):
        # Read the output file
        with open(file_path, 'r') as f:
            results = [json.loads(line) for line in f if line.strip()]

        # Count the number of correct answers
        if args.dataset == 'math':
            correct_answers = 0
            for item in results:
            # Use compare_answers with the already parsed answer from convert_to_json
                is_correct = compare_answers(item['input'], item['label'], item['answer'])
                correct_answers += int(is_correct)
        else:
            correct_answers = sum(1 for item in results if item['label'] == item['answer'])
        
        total_examples = len(results)
        accuracy = (correct_answers / total_examples) * 100
        
        print(f"Total number of correct answer is {correct_answers}/{total_examples}")
        print(f"The accuracy is {accuracy:.2f}%")

        # Print total and average token counts
        total_tokens = sum(item['token_count'] for item in results)
        avg_tokens = total_tokens / total_examples
        print(f"Total number of tokens in the generated output: {total_tokens}")
        print(f"Average number of tokens per generated output: {avg_tokens:.2f}")

    if args.accelerate:
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for evaluating models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for generation (e.g., 'gsm8k').")
    parser.add_argument("--prompt", type=str, required=True, choices=["direct", "zero-shot", "few-shot"], help="Prompt type to use.")
    parser.add_argument("--prompt_system", type=str, default="irpo", choices=["irpo", "concise", "hand1", "hand2", "hand3", "hand4", "no"], 
                        help="Style of system prompt to use for evaluation")
    parser.add_argument("--few_shot_path", type=str, help="Path to the exemplar file for few-shot prompting")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation.")
    parser.add_argument("--accelerate", action="store_true", help="Whether to use distributed generation.")
    parser.add_argument("--flash_attention", action="store_true", help="Whether to use flash attention.")
    args = parser.parse_args()

    evaluate(args)
