#!/bin/bash

# Function to check if a port is available
check_port() {
    local port=$1
    nc -z localhost $port >/dev/null 2>&1
    [ $? -ne 0 ]
}

# Function to get a random available port
get_random_port() {
    local min_port=29000
    local max_port=32000
    local port
    
    while true; do
        # Generate random port number within range
        port=$(shuf -i $min_port-$max_port -n 1)
        
        # Check if port is available
        if check_port $port; then
            echo $port
            return 0
        fi
    done
}

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_path> <dataset_type> [batch_size]"
    exit 1
fi

# Store command line arguments
MODEL_PATH="$1"
DATASET_TYPE="$2"
BATCH_SIZE=${3:-64}  # Default to 64 if not provided

# Convert path to lowercase for case-insensitive matching
LOWER_PATH=$(echo "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')

# Function to extract model name from path
get_model_name() {
    local path="$1"
    
    # Use grep to find the base model name pattern
    if [[ $path =~ (llama[^/]+|qwen[^/]+|gemma[^/]+|deepseek[^/]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "ERROR: Could not extract model name from path: $MODEL_PATH"
        exit 1
    fi
}

# Get the model name
MODEL_NAME=$(get_model_name "$LOWER_PATH")
echo "Model Name: $MODEL_NAME"

# Set max tokens based on dataset type
if [ "$DATASET_TYPE" = "math" ]; then
    MAX_TOKENS=1024
else
    MAX_TOKENS=512
fi

# Get a random available port
PORT=$(get_random_port)
echo "Using port: $PORT"

# Run the evaluation command
accelerate launch --main_process_port $PORT src/evaluation.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET_TYPE" \
    --prompt direct \
    --prompt_system no \
    --max_new_tokens "$MAX_TOKENS" \
    --batch_size "$BATCH_SIZE" \
    --accelerate