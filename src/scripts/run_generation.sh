#!/bin/bash

# Environment settings
CUDA_DEVICES="4,5,6,7"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# Base paths
BASE_MODEL_DIR="./models/"
BASE_FEW_SHOT_DIR="./data/few_shot_examples"
LOG_DIR="logs/generation"

# Model configurations
# Format: "model_name:batch_size"
declare -a model_configs=(
    "llama-3.2-1b-instruct:64"
    "llama-3.2-3b-instruct:32"
    "qwen2.5-math-1.5b-instruct:32"
    "qwen2.5-3b-instruct:32"
    "gemma-2-2b-it:32"
    "llama-3.1-8b-instruct:32"
    "deepseek-math-7b-instruct:32"
)

# Prompt system configurations
# Zero-shot prompt systems
declare -a zero_shot_prompt_systems=(
    "irpo"
)

# Few-shot examples types
declare -a few_shot_prompt_systems=(
    "gpt4o"         
    "original-cot"  
    "irpo"       
)

# Dataset configurations
declare -a datasets=(
    "gsm8k" 
    "math"
)

# Prompt types to run
declare -a prompt_types=(
    "few-shot"
    "zero-shot"
)

# Create log directory
mkdir -p "$LOG_DIR"

# Log file
LOG_FILE="$LOG_DIR/generation.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to get few-shot path based on configuration
get_few_shot_path() {
    local model_name=$1
    local dataset=$2
    local prompt_system_key=$3
    
    # For gpt4o and original-cot, use dataset-specific paths
    if [[ "$prompt_system_key" == "gpt4o" || "$prompt_system_key" == "original-cot" ]]; then
        echo "${BASE_FEW_SHOT_DIR}/${dataset}/few-shot-${prompt_system_key}.json"
    else
        # For irpo and short, use model and dataset specific paths
        echo "${BASE_FEW_SHOT_DIR}/filtered_few_shot_prompts/${dataset}/${model_name}/${prompt_system_key}_128_128.json"
    fi
}

# Create log file
touch "$LOG_FILE"
log_message "Starting generation across configurations"

# Calculate total combinations
total_few_shot=$((${#few_shot_prompt_systems[@]} * ${#model_configs[@]} * ${#datasets[@]}))
total_zero_shot=$((${#zero_shot_prompt_systems[@]} * ${#model_configs[@]} * ${#datasets[@]}))
total_combinations=$((total_few_shot + total_zero_shot))
current=0

# Loop through each prompt type
for prompt_type in "${prompt_types[@]}"; do
    log_message "Starting processing for prompt type: $prompt_type"
    
    if [ "$prompt_type" = "zero-shot" ]; then
        # For zero-shot, use zero-shot prompt systems
        prompt_systems=("${zero_shot_prompt_systems[@]}")
    else
        # For few-shot, use few-shot prompt systems
        prompt_systems=("${few_shot_prompt_systems[@]}")
    fi
    
    # Loop through appropriate prompt systems
    for prompt_system_key in "${prompt_systems[@]}"; do
        log_message "Starting processing for prompt system key: $prompt_system_key"
        
        # Loop through each dataset
        for dataset in "${datasets[@]}"; do
            # Loop through each model configuration
            for model_config in "${model_configs[@]}"; do
                # Split model configuration
                IFS=':' read -r model_name batch_size <<< "$model_config"
                
                # Construct model path
                model_path="${BASE_MODEL_DIR}/${model_name}"
                
                ((current++))
                log_message "Starting combination $current/$total_combinations"
                log_message "Prompt Type: $prompt_type, Prompt System Key: $prompt_system_key, Dataset: $dataset, Model: $model_name"
                
                # Set base command
                cmd="python src/reasoning_generation.py \
                    --prompt $prompt_type \
                    --model_path \"$model_path\" \
                    --model_name \"$model_name\" \
                    --dataset \"$dataset\" \
                    --do_sample \
                    --batch_size \"$batch_size\" \
                    --num_diverse_path 16 \
                    --use_vllm"

                # Add prompt system for zero-shot (always IRPO)
                if [ "$prompt_type" = "zero-shot" ]; then
                    cmd="$cmd --prompt_system $prompt_system_key"
                fi

                # Add few-shot path for few-shot
                if [ "$prompt_type" = "few-shot" ]; then
                    few_shot_path=$(get_few_shot_path "$model_name" "$dataset" "$prompt_system_key")
                    
                    # Check if few-shot path exists
                    if [ ! -f "$few_shot_path" ]; then
                        log_message "Warning: Few-shot path does not exist: $few_shot_path"
                        continue
                    fi
                    
                    cmd="$cmd --few_shot_path \"$few_shot_path\" --prompt_system_key \"$prompt_system_key\""
                    cmd="$cmd --prompt_system no"
                fi
                
                # Set max_new_tokens based on dataset
                if [ "$dataset" = "math" ]; then
                    cmd="$cmd --max_new_tokens 1024"
                else
                    cmd="$cmd --max_new_tokens 512"
                fi

                # Run generation
                eval "$cmd" 2>&1 | tee "$LOG_DIR/${prompt_type}_${prompt_system_key}_${dataset}_${model_name}.txt"
                
                # Check if generation was successful
                if [ $? -eq 0 ]; then
                    log_message "Successfully completed generation for $prompt_type, prompt system: $prompt_system_key, model: $model_name on $dataset"
                else
                    log_message "Error in generation combination $current/$total_combinations"
                fi
                
                # Optional: Add a delay between runs
                sleep 3
            done
        done
        
        log_message "Completed all combinations for prompt system key: $prompt_system_key"
    done
    
    log_message "Completed all combinations for prompt type: $prompt_type"
done

log_message "Completed all generation combinations"