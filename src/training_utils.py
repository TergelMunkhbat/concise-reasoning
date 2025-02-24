import re
import json
import torch
import torch.distributed as dist
import random
from typing import Dict, List, Tuple
from collections import defaultdict
from vllm import LLM, SamplingParams
from tqdm import tqdm

from prompt import SYSTEM_PROMPT_IRPO, SYSTEM_PROMPT_CONCISE, SYSTEM_PROMPT_COMPACT
from prompt import SYSTEM_PROMPT_SUMMARY, SYSTEM_PROMPT_SHORT, SYSTEM_PROMPT_SHORT2
from prompt import ZERO_SHOT_PROMPT
        
        
def tokenize_function(inputs, tokenizer, batch_size, device) -> list:
    """
    Tokenize the inputs

    Parameters:
        inputs (list): list of inputs.
        tokenizer (AutoTokenizer): Tokenizer object.
        batch_size (int): Size of each batch
    
    Returns:
        list: Tokenized batches
    """
    batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
    batches_tok = []
    
    # Store original settings
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    try:
        for input_batch in batches:
            batches_tok.append(
                tokenizer(
                    input_batch,
                    return_tensors='pt',
                    padding='longest',
                    truncation=False,
                    pad_to_multiple_of=8,
                    add_special_tokens=False,
                ).to(device)
            )
    finally:
        # Restore original settings
        tokenizer.padding_side = original_padding_side
    
    return batches_tok


def generate_text(model, tokenizer, prompts_tokenized, args) -> list:
    """
    Generate text based on the prompt using the provided model and tokenizer.

    Parameters:
        model (AutoModelForCausalLM): Pretrained model.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the model.
        prompts_tokenized (dict): Tokenized prompts.
        args (argparse.Namespace): Parsed command-line arguments.
    
    Returns:
        tuple: A tuple containing:
            - list: Generated texts.
            - list: Number of tokens in each generated output.
    """
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    do_sample = args.do_sample
    input_length = len(prompts_tokenized['input_ids'][0])

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Generate output ids based on the input ids
        output_tokenized = model.generate(**prompts_tokenized,
                                          max_new_tokens=max_new_tokens,
                                          temperature=temperature,
                                          top_k=top_k,
                                          top_p=top_p,
                                          do_sample=do_sample)
        
        # Calculate the number of generated tokens for each output
        generated_tokens = []
        for output in output_tokenized:
            # Find where the actual content ends
            end_index = len(output)
            for i in range(input_length, len(output)):
                if output[i] == tokenizer.eos_token_id:
                    end_index = i
                    break
            
            # Count only the actual generated tokens
            generated_tokens.append(end_index - input_length)

        # Decode the generated output ids into text, skipping special tokens
        outputs = tokenizer.batch_decode(output_tokenized[:, input_length:], skip_special_tokens=True)
    
    return outputs, generated_tokens


def format_zero_shot_prompt(example: dict, system_style: str, model_name: str) -> dict:
    """
    Format the example using model chat template.
    
    Parameters:
        example (dict): Dictionary containing input.
        system_style (str): Style of system prompt to use
    
    Returns:
        dict: Formatted input.
    """
    # Select system prompt based on style
    system_prompt = {
        "irpo": SYSTEM_PROMPT_IRPO,
        "concise": SYSTEM_PROMPT_CONCISE,
        "hand1": SYSTEM_PROMPT_SUMMARY,
        "hand2": SYSTEM_PROMPT_COMPACT,
        "hand3": SYSTEM_PROMPT_SHORT,
        "hand4": SYSTEM_PROMPT_SHORT2
    }.get(system_style)
    
    # Format the user message
    user_message = ZERO_SHOT_PROMPT.format(question=example["input"])
    
    # Create messages in chat format
    if "gemma" in model_name:
        user_message = system_prompt + "\n" + user_message
        messages = [
            {"role": "user", "content": user_message}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    
    return {'messages': messages}


def format_few_shot_prompt(example: dict, exemplar_file: str) -> dict:
    """
    Format the example into chat messages format with random exemplars of the same type.
    
    Parameters:
        example (dict): Dictionary containing an input question and dataset type
        exemplar_file (str): Path to the file containing shortest solutions
    
    Returns:
        dict: Formatted input in chat format
    """
    # Load examplers from file
    exemplars = load_examplars(exemplar_file)
    
    messages = []
    
    # Get random exemplars of the same type
    exemplars = get_random_examplars(example, exemplars)
    
    # Add exemplar conversations
    for exemplar in exemplars:
        # Add user message with question
        messages.append({
            "role": "user",
            "content": f"Question: {exemplar['question']}\nSolution:"
        })
        
        # Add assistant response with short solution
        messages.append({
            "role": "assistant",
            "content": exemplar['solution']
        })
    
    # Add the current question as the final user message
    messages.append({
        "role": "user",
        "content": f"Question: {example['input']}\nSolution:"
    })
    
    return {'messages': messages}


def parse_answer(string, type) -> str:
    """
    Parse and extract answer from a string

    Parameters:
        string (str): The input string to parse
    
    Return:
        str: A string containing the parsed answer, or empty if the string does not match the expected format
    """
    # Split the string based on the phrase "Final answer:"
    string_parts = string.split("answer is")
    # Set a flag to indicate if there is an answer
    answer_flag = True if len(string_parts) > 1 else False
    # Get the last part of the string
    string = string_parts[-1]
    
    # Remove comas from the string
    string = string.replace(",", "")
    
    if type == "gsm8k":
        # Find all numeric values in the string using regular expression
        string = [s for s in re.findall(r'-?\d+\.?\d*', string)]

    # If there are no candidates in the list, set string to an empty string
    if len(string) == 0:
        string = ""
    else:
        # Choose the first or last element based on the answer flag
        if answer_flag:
            string = string[0]
        else:
            string = string[-1]
    
    # Remove trailing period if present
    string = string.rstrip('.')
    string = string.replace("\n", "")
    # Remove trailing zeros from decimals
    if re.match(r'^-?\d+\.\d+$', string):
        string = string.rstrip('0').rstrip('.')
    
    return string


def safe_gather_outputs(outputs, accelerator):
    """
    Gather on CPU memory instead of GPU memory to avoid OOM errors.
    """
    accelerator.wait_for_everyone()

    my_rank = accelerator.process_index
    world_size = accelerator.num_processes

    if my_rank != 0:
        # PyTorch 2.0 only has send_object_list, so wrap your data in a list
        dist.send_object_list([outputs], dst=0)
        return []
    else:
        gathered = []
        # Add my own outputs
        gathered.extend(outputs)

        for src_rank in range(1, world_size):
            tmp_list = [None]
            dist.recv_object_list(tmp_list, src=src_rank)
            # tmp_list[0] is the actual object from that rank
            gathered.extend(tmp_list[0])

        return gathered

def generate_responses(datasets, model, tokenizer, args, accelerator=None) -> list:
    """
    Process datasets and generate text using the given model and tokenizer.

    Parameters:
        datasets (Dataset): original dataset.
        model (AutoModelForCausalLM): Pretrained model.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the model.
        args (argparse.Namespace): Parsed command-line arguments.
        accelerator (Accelerator, optional): Accelerator for distributed generate text, if any.

    Return:
        tuple: A tuple containing:
            - list: A list of generated texts for each input example in the dataset.
            - list: A list of token counts for each generated output.
    """
    outputs = []
    output_token_counts = []

    if accelerator:
        with accelerator.split_between_processes(datasets["prompt"]) as dataset_splitted:
            tokenized_datasets = tokenize_function(dataset_splitted, tokenizer, args.batch_size, "cuda")
            for batch in tqdm(tokenized_datasets, desc="Generating reasoning paths"):
                batch_outputs, batch_tokens = generate_text(model, tokenizer, batch, args)
                outputs.append(batch_outputs)
                output_token_counts.extend(batch_tokens)

        torch.cuda.empty_cache()

        if accelerator.is_local_main_process:
            # rank 0 will end up with everything
            gathered_outputs = safe_gather_outputs(outputs, accelerator)
            gathered_token_counts = safe_gather_outputs(output_token_counts, accelerator)
        else:
            # rank>0 returns empty
            _ = safe_gather_outputs(outputs, accelerator)
            _ = safe_gather_outputs(output_token_counts, accelerator)
            gathered_outputs, gathered_token_counts = [], []

        return gathered_outputs, gathered_token_counts
    else:
        tokenized_datasets = tokenize_function(datasets["prompt"], tokenizer, args.batch_size, "cuda")
        for batch in tqdm(tokenized_datasets, desc="Generating reasoning paths"):
            batch_outputs, batch_tokens = generate_text(model, tokenizer, batch, args)
            outputs.append(batch_outputs)
            output_token_counts.extend(batch_tokens)
        return outputs, output_token_counts
    

def load_examplars(file_path: str) -> Dict[str, List[Dict]]:
    """Load and organize exemplars by dataset type."""
    with open(file_path, 'r') as file:
        exemplars = json.load(file)
    
    # Organize exemplars by dataset type
    exemplars_by_type = defaultdict(list)
    for ex in exemplars:
        exemplars_by_type[ex['dataset']].append(ex)

    return exemplars_by_type


def get_random_examplars(example: dict, exemplars: Dict[str, List[Dict]], num_shots: int = 8) -> List[Dict]:
    """Get random exemplars of the same type as the example."""
    dataset_type = example['dataset']
    available_exemplars = exemplars[dataset_type]
    
    if len(available_exemplars) < num_shots:
        print(f"Warning: Not enough exemplars for dataset type {dataset_type}.")
        return available_exemplars
    
    return random.sample(available_exemplars, num_shots)


def generate_text_vllm(model_path: str, prompts: List[str], args) -> Tuple[List[str], List[int]]:
    """
    Generate text using vLLM for improved performance.
    
    Parameters:
        model_path (str): Path to the model checkpoint
        prompts (List[str]): List of input prompts
        args: Generation arguments containing:
            - max_new_tokens (int): Maximum number of new tokens to generate
            - temperature (float, optional): Sampling temperature
            - top_k (int, optional): Top-k sampling parameter
            - top_p (float, optional): Top-p sampling parameter
            - do_sample (bool): Whether to use sampling for generation
    
    Returns:
        tuple: A tuple containing:
            - list: Generated texts
            - list: Number of tokens in each generated output
            
    Raises:
        RuntimeError: If vLLM initialization fails
    """
    try:
        # Initialize vLLM with error handling
        llm = LLM(
            model=model_path,
            enable_prefix_caching=True,
            dtype="bfloat16",
            tensor_parallel_size=torch.cuda.device_count(),
        )
        
        # Set up sampling parameters with proper defaults
        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature if args.do_sample else 0.0,
            top_k=args.top_k if args.do_sample else -1,
            top_p=args.top_p if args.do_sample else 1.0,
        )
        
        # Generate outputs with batch processing
        outputs = llm.generate(prompts, sampling_params)
        
        # Extract generated texts and token counts
        generated_texts = []
        token_counts = []
        
        for output in outputs:
            generated_texts.append(output.outputs[0].text)
            token_counts.append(len(output.outputs[0].token_ids)-1)
            
        return [generated_texts], token_counts
        
    except Exception as e:
        raise RuntimeError(f"vLLM generation failed: {str(e)}")
    finally:
        # Clean up resources
        if 'llm' in locals():
            del llm
            torch.cuda.empty_cache()


def get_generator(args):
    """
    Get the appropriate text generation function based on args
    
    Parameters:
        args: Arguments containing generation parameters and flags
        
    Returns:
        Callable: Function that takes (prompts, model, tokenizer, args) for standard
                 generation or (prompts, model_path, args) for vLLM
    """
    if args.use_vllm:
        return generate_text_vllm
    else:
        return generate_text
