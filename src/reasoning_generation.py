import os
import random
import argparse
from datasets import Dataset
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from transformers import AutoTokenizer

from dataset import GSM8kDatasetLoader, MATHDatasetLoader
from training_utils import format_zero_shot_prompt, generate_responses, format_few_shot_prompt, get_generator
from utils import get_config_dir, convert_to_json
from model import load_model, load_model_with_flash_attention
from prompt import LLAMA_CHAT_TEMPLATE


def generate(args) -> None:
    """
    Generate reasoning paths and save to JSON

    Parameters:
        args (argparse.Namespace) Parsed command-line arguments.
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
    datasets = datasets['train']
    
    if args.num_first_samples:
        datasets = datasets.select(range(args.num_first_samples))
    elif args.num_random_samples:
        random.seed(42)
        selected_indices = random.sample(range(len(datasets)), args.num_random_samples)
        datasets = datasets.select(selected_indices)
    
    # Create output directory
    output_dir = get_config_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # Split dataset into chunks
    chunk_size = 32000
    num_chunks = (len(datasets) + chunk_size - 1) // chunk_size
    
    # Get the appropriate generator function
    generator_fn = get_generator(args)

    # Load model and tokenizer if not using vLLM
    model = None
    tokenizer = None
    accelerator = None
    
    if not args.use_vllm:
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
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Format prompts based on whether using direct input or chat template
    def format_prompt(example):
        if args.prompt == "direct":
            if "qwen" in args.model_name:
                formatted_prompt = example['input']
            else:
                formatted_prompt = tokenizer.bos_token + example['input']
        else:
            if "llama" in args.model_name:
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
        if args.use_vllm:    
            # Remove bos token if it exists at the start for vllm
            if tokenizer.bos_token and formatted_prompt.startswith(tokenizer.bos_token):
                        formatted_prompt = formatted_prompt[len(tokenizer.bos_token):]
        return {"prompt": formatted_prompt}
    
    # Greedy decoding
    if not args.do_sample:
        args.temperature = None
        args.top_p = None
        args.top_k = None
        
    for i in range(num_chunks):

        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(datasets))

        # Select chunch datasets
        chunk_datasets = datasets.select(indices=range(start_idx, end_idx))

        # Repeat examples for diverse path decoding
        chunk_datasets = [[row] * args.num_diverse_path for row in chunk_datasets]
        chunk_datasets = [item for sublist in chunk_datasets for item in sublist]
        chunk_datasets = Dataset.from_list(chunk_datasets)
        
        if args.prompt != "direct":
            # Apply preprocessing for this iteration
            if args.prompt == "zero-shot":
                chunk_datasets = chunk_datasets.map(lambda x: format_zero_shot_prompt(x, args.prompt_system, args.model_name))
            elif args.prompt == "few-shot":
                chunk_datasets = chunk_datasets.map(lambda x: format_few_shot_prompt(x, args.few_shot_path))
        
        chunk_datasets = chunk_datasets.map(format_prompt)
        if "messages" in chunk_datasets.column_names:
            chunk_datasets = chunk_datasets.remove_columns(['messages'])

        if args.use_vllm:
            outputs, output_token_counts = generator_fn(
                args.model_path,
                chunk_datasets["prompt"],
                args
            )
        else:
            outputs, output_token_counts = generate_responses(chunk_datasets, model, tokenizer, args, accelerator)

        # Save to JSON
        if not args.accelerate or (args.accelerate and accelerator.is_main_process):
            convert_to_json(chunk_datasets, outputs, output_token_counts, output_dir, model_name=args.model_name, chunk_index=i)
    
    if args.accelerate:
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for generating reasoning paths")
    parser.add_argument("--prompt", type=str, required=True, choices=["direct", "zero-shot", "few-shot", "no"], help="Prompt type to use")
    parser.add_argument("--prompt_system", type=str, default="irpo", choices=["irpo", "concise", "hand1", "hand2", "hand3", "hand4", "no"],
                        help="Style of system prompt to use for evaluation")
    parser.add_argument("--prompt_system_key", type=str, default=None,
                        help="Override prompt_system in the output directory template (used to differentiate output "
                             "directory for few-shot prompting, where prompt system is not used)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for generation (e.g., 'gsm8k').")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Softmax temperature for sampling.")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k value for sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p value for sampling.")
    parser.add_argument("--do_sample", action='store_true', 
                        help="Enable sampling for diverse generation; uses temperature, top_k, and top_p settings.")
    parser.add_argument("--num_diverse_path", type=int, default=1, help="Number of diverse decoding paths.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation.")
    parser.add_argument("--accelerate", action="store_true", help="Whether to use distributed generation.")
    parser.add_argument("--flash_attention", action="store_true", help="Whether to use flash attention.")
    parser.add_argument("--few_shot_path", type=str, 
                        help="Path to the JSON file containing few-shot exemplars (required when prompt='few-shot')")
    parser.add_argument("--num_first_samples", type=int, default=None, help="Use first N samples from dataset")
    parser.add_argument("--num_random_samples", type=int, default=None, help="Use random N samples from dataset")
    parser.add_argument("--use_vllm", action="store_true", help="Whether to use vLLM for generation.")
    args = parser.parse_args()

    generate(args)

