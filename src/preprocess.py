import json
import os
import ast
import math
import argparse
import numpy as np
from pathlib import Path


def process_file(filename):
    """Read and parse JSON lines from a file."""
    with open(filename, 'r') as file:
        return [json.loads(line) for line in file if line.strip()]


def compare_answers(question, parsed_label, parsed_answer):
    
    # Convert to lists if they're string representations of lists
    if isinstance(parsed_label, str):
        parsed_label = ast.literal_eval(parsed_label)
    if isinstance(parsed_answer, str):
        parsed_answer = ast.literal_eval(parsed_answer)
    
    if 'separated by commas' in question:
        # Convert each sublist to a frozen set for hashability
        answer_sets = {frozenset(str(x) for x in ans) for ans in parsed_answer}
        label_sets = {frozenset(str(x) for x in lab) for lab in parsed_label}
        # Check for any intersection between answer_sets and label_sets
        return bool(answer_sets & label_sets)
    else:
        # Convert lists to sets for efficient membership testing
        return bool(set(str(x) for x in parsed_answer) & set(str(x) for x in parsed_label))
    

def filter_rationales(data, dataset, max_tokens, min_tokens=0, only_correct=True):
    # Step 1: Filter by correctness
    correctness_filtered = []
    if only_correct:
        if dataset == "gsm8k":
            correctness_filtered = [item for item in data if item['answer'] == item['label']]
        elif dataset == "math":
            correctness_filtered = [
                item for item in data 
                if compare_answers(
                    item['input'],
                    item['label'],
                    item['answer']
                )
            ]
    else:
        correctness_filtered = data
    
    # Step 2: Remove duplicates efficiently
    seen_rationales = set()
    unique_data = []
    duplicates_removed = 0

    for item in correctness_filtered:
        if item['rationale'] not in seen_rationales:
            seen_rationales.add(item['rationale'])
            unique_data.append(item)
        else:
            duplicates_removed += 1
    
    # Step 3: Filter by token count
    removed_by_min_tokens = len([item for item in unique_data if item['token_count'] < min_tokens])
    removed_by_max_tokens = len([item for item in unique_data if item['token_count'] >= max_tokens])
    filtered_rationales = [item for item in unique_data 
                         if min_tokens <= item['token_count'] < max_tokens]
    
    # Calculate token statistics
    token_counts = [r['token_count'] for r in filtered_rationales]
    
    print(f"Filtered Rationales Statistics:")
    print(f"Original number of rationales: {len(data)}")
    if only_correct:
        print(f"Number after correctness filtering: {len(correctness_filtered)} ({len(correctness_filtered)/len(data)*100:.2f}%)")
        print(f"Number after removing duplicates: {len(unique_data)} (removed {duplicates_removed} duplicates)")
    print(f"Number removed due to min_tokens={min_tokens}: {removed_by_min_tokens} ({removed_by_min_tokens/len(unique_data)*100:.2f}% of deduplicated ones)")
    print(f"Number removed due to max_tokens={max_tokens}: {removed_by_max_tokens} ({removed_by_max_tokens/len(unique_data)*100:.2f}% of deduplicated ones)")
    print(f"Final number of rationales: {len(filtered_rationales)} ({len(filtered_rationales)/len(data)*100:.2f}%)")
    print(f"Total number of removed rationales: {len(data) - len(filtered_rationales)} ({(len(data) - len(filtered_rationales))/len(data)*100:.2f}%)")
    
    if filtered_rationales:  # Only calculate statistics if we have rationales
        print(f"\nToken Statistics for Filtered Rationales:")
        print(f"Average token count: {np.mean(token_counts):.2f}")
        print(f"Median token count: {np.median(token_counts):.2f}")
        print(f"Standard deviation of token counts: {np.std(token_counts):.2f}")
        print(f"Minimum token count: {min(token_counts)}")
        print(f"Maximum token count: {max(token_counts)}")
    else:
        print("\nNo rationales remained after filtering.")
    
    return filtered_rationales


def save_filtered_rationales(rationales, max_rows_per_file=32000, dir_path='./'):
    """Save rationales to JSON files with specified maximum rows per file."""
    os.makedirs(dir_path, exist_ok=True)
    num_files = math.ceil(len(rationales) / max_rows_per_file)
    
    for i in range(num_files):
        start_idx = i * max_rows_per_file
        end_idx = min((i + 1) * max_rows_per_file, len(rationales))
        
        data_to_save = [
            {
                'input': rationale['input'],
                'label': rationale['label'],
                'rationale': rationale['rationale'],
                'token_count': rationale['token_count'],
                'dataset': rationale['dataset'],
                '_id': rationale['_id']
            }
            for rationale in rationales[start_idx:end_idx]
        ]
        
        filename = f'{dir_path}/filtered_correct_rationales_{i}.json'
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"Saved {len(data_to_save)} rationales to '{filename}'")


def preprocess(args) -> None:
    """Preprocess rationales from JSON files."""
    if args.dataset == "gsm8k":
        max_tokens = 512
    elif args.dataset == "math":
        max_tokens = 1024

    data_dir = Path(args.data_dir)
    output_dir = data_dir / 'train'
    
    # Ensure necessary directories exist
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = [item for file in data_dir.glob('*.json') for item in process_file(file)]
    
    filtered_correct = filter_rationales(all_data, args.dataset, max_tokens)
    save_filtered_rationales(filtered_correct, dir_path=output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for preprocessing rationales")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing JSON files.")
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "math"], help="Dataset to preprocess rationales for.")
    args = parser.parse_args()

    preprocess(args)