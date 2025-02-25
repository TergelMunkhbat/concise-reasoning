import json
import re
import os

def process_file(filename):
    with open(filename, 'r') as file:
        return [json.loads(line) for line in file if line.strip()]
    
# Function to add the parsed budget and check patterns
def check_patterns(data):
    budget_pattern_count = 0
    
    for item in data:
        rationale = item.get('rationale', '')
        rationale = rationale.replace(',', '')
        
        if "Budget:" in rationale:
            budget_pattern_count += 1

    return budget_pattern_count


def add_parsed_budget(data):
    for item in data:
        rationale = item.get('rationale', '')
        rationale = rationale.replace(',', '')  # Remove commas
        
        # Match `Budget: <number>` or `Budget: [[<number>]]`
        match = re.search(r'Budget:\s*(\[\[(\d+)\]\]|\d+)', rationale)
        
        if match:
            # If the budget is inside brackets, extract the second group
            budget = match.group(2) if match.group(2) else match.group(1)
            item['budget_estimate'] = str(int(budget))
        else:
            # Fallback: find the last number in the rationale
            last_number_match = re.findall(r'\d+', rationale)
            item['budget_estimate'] = int(last_number_match[-1]) if last_number_match else None
        item['budget_estimate'] = str(item['budget_estimate'])
        del item['rationale']
        del item['answer']
        del item['token_count']
    
    return data


def main(args):
    dir = f"./data/{args.dataset}/{args.model_name}/results/budget_estimation"
    # get a json file starting with "output" from the directory
    files = [f for f in os.listdir(dir) if re.match(r'output.*\.json', f)]
    files.sort()

    path = files[-1]
    path = os.path.join(dir, path)

    output_data = process_file(path)

    # Apply the function
    budget_count = check_patterns(output_data)

    # Display results
    print(f"Entries with 'Budget:' pattern: {budget_count}")

    updated_data = add_parsed_budget(output_data)

    # save input data to a json file
    output_path = f"{dir}/token_limit_input.json"
    with open(output_path, 'w') as file:
        json.dump(updated_data, file, indent=4)


if __name__ == "__main__":
    # parse args
    import argparse
    parser = argparse.ArgumentParser(description='Process budget estimation output')
    parser.add_argument('--dataset', type=str, default='math', help='Dataset to process')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    args = parser.parse_args()

    main(args)