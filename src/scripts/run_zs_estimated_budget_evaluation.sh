#!/bin/bash

# Environment settings
CUDA_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# Base paths
BASE_MODEL_DIR="./models"
LOG_DIR="logs/evaluation"

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
    "zero-shot"
)

# Create log directory
mkdir -p "$LOG_DIR"

# Log file
LOG_FILE="$LOG_DIR/evaluation.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Create log file
touch "$LOG_FILE"
log_message "Starting evaluation across configurations"

# Calculate total combinations
total_combinations=$((${#model_configs[@]} * ${#datasets[@]}))
current=0
        
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
        
        # Step 1: Estimate the budget
        log_message "Prompt Type: zero-shot, Prompt System Key: budget_estimation, Dataset: $dataset, Model: $model_name"

        # Set base command
        cmd="python src/evaluation.py \
            --model_path \"$model_path\" \
            --model_name \"$model_name\" \
            --dataset \"$dataset\" \
            --prompt \"zero-shot\" \
            --prompt_system \"budget_estimation\" \
            --batch_size \"$batch_size\" \
            --use_vllm"
        
        # Set max_new_tokens based on dataset
        if [ "$dataset" = "math" ]; then
            cmd="$cmd --max_new_tokens 1024"
        else
            cmd="$cmd --max_new_tokens 512"
        fi

        # Run evaluation
        eval "$cmd" 2>&1 | tee "$LOG_DIR/zero-shot_budget_estimation_${dataset}_${model_name}.txt"
        
        # Check if evaluation was successful
        if [ $? -eq 0 ]; then
            log_message "Successfully completed evaluation for zero-shot, prompt system: budget_estimation, model: $model_name on $dataset"
        else
            log_message "Error in evaluation prompt system: budget_estimation, combination $current/$total_combinations"
        fi

        # Step 2: Parse the budget
        python src/estimated_budget_process.py \
                --model_name $model_name \
                --dataset $dataset

        # Step 3: Run the evaluation on the estimated budget
        log_message "Prompt Type: zero-shot, Prompt System Key: estimated_budget, Dataset: $dataset, Model: $model_name"

        # Set base command
        cmd="python src/evaluation.py \
            --model_path \"$model_path\" \
            --model_name \"$model_name\" \
            --dataset \"$dataset\" \
            --prompt \"zero-shot\" \
            --prompt_system \"estimated_budget\" \
            --estimated_budget_data_path \"data/$dataset/$model_name/results/budget_estimation/token_limit_input.json\" \
            --batch_size \"$batch_size\" \
            --use_vllm"
        
        # Set max_new_tokens based on dataset
        if [ "$dataset" = "math" ]; then
            cmd="$cmd --max_new_tokens 1024"
        else
            cmd="$cmd --max_new_tokens 512"
        fi

        # Run evaluation
        eval "$cmd" 2>&1 | tee "$LOG_DIR/zero-shot_estimated_budget_${dataset}_${model_name}.txt"
        
        # Check if evaluation was successful
        if [ $? -eq 0 ]; then
            log_message "Successfully completed evaluation for zero-shot, prompt system: estimated_budget, model: $model_name on $dataset"
        else
            log_message "Error in evaluation prompt system: estimated_budget, combination $current/$total_combinations"
        fi
        
        # Optional: Add a delay between runs
        sleep 3
    done
done

log_message "Completed all evaluation combinations"