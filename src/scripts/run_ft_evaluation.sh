#!/bin/bash

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

# Run the evaluation command
python src/evaluation.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET_TYPE" \
    --prompt direct \
    --prompt_system no \
    --max_new_tokens "$MAX_TOKENS" \
    --batch_size "$BATCH_SIZE" \
    --use_vllm