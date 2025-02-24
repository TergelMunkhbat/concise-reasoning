#!/bin/bash

# Environment settings
CUDA_DEVICES="4,5,6,7"

# Model configurations
# Format: "model_name"
declare -a model_configs=(
    "llama-3.2-1b-instruct"
    "llama-3.2-3b-instruct"
    "qwen2.5-math-1.5b-instruct"
    "qwen2.5-3b-instruct"
    "gemma-2-2b-it"
    "llama-3.1-8b-instruct"
    "deepseek-math-7b-instruct"
)

# Dataset configurations
declare -a datasets=(
    "gsm8k"
    "math"
)

# Base paths
BASE_MODEL_DIR="./models"
BASE_OUTPUT_DIR="./models/trained"

# Log file
LOG_FILE="training_automation.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run a single iteration of generation or training
run_iteration() {
    local model_name=$1
    local model_path=$2
    local dataset=$3
    local iteration=$4
    local mode=$5
    local max_new_tokens=$6
    local batch_size=$7
    local grad_steps=$8
    local output_dir=$9

    if [ "$mode" = "generate" ]; then
        log_message "Starting distributed generation phase for iteration $iteration"
        # Run with accelerator for generation
        accelerate launch --main_process_port 29506 src/training.py \
            trainer.model_name=$model_name \
            trainer.model_path=$model_path \
            trainer.dataset=$dataset \
            trainer.type="best_reward" \
            trainer.max_new_tokens=$max_new_tokens \
            trainer.batch_size=$batch_size \
            trainer.grad_steps=$grad_steps \
            trainer.output_dir=$output_dir \
            env.CUDA_VISIBLE_DEVICES=\"$CUDA_DEVICES\" \
            trainer.accelerate=True \
            2>&1 | tee -a "$LOG_FILE"
    else
        log_message "Starting single process training phase for iteration $iteration"
        # Run without accelerator for training
        python src/training.py \
            trainer.model_name=$model_name \
            trainer.model_path=$model_path \
            trainer.dataset=$dataset \
            trainer.type="best_reward" \
            trainer.max_new_tokens=$max_new_tokens \
            trainer.batch_size=$batch_size \
            trainer.grad_steps=$grad_steps \
            trainer.output_dir=$output_dir \
            env.CUDA_VISIBLE_DEVICES=\"$CUDA_DEVICES\" \
            trainer.accelerate=False \
            2>&1 | tee -a "$LOG_FILE"
    fi
}

# Create log file
touch "$LOG_FILE"
log_message "Starting training automation"

# Counter for tracking progress
total_combinations=$((${#model_configs[@]} * ${#datasets[@]}))
current=0

# Loop through each configuration
for dataset in "${datasets[@]}"; do
    for config in "${model_configs[@]}"; do
        # Split configuration
        IFS=':' read -r model_name <<< "$config"
        model_path="${BASE_MODEL_DIR}/${model_name}"
        
        ((current++))
        log_message "Starting combination $current/$total_combinations"
        log_message "Model: $model_name, Dataset: $dataset"
        
        # Set dataset-specific parameters
        if [ "$dataset" = "gsm8k" ]; then
            max_new_tokens=512
            batch_size=16
            grad_steps=1
            eval_batch_size=32
        else
            max_new_tokens=1024
            batch_size=8
            grad_steps=2
            eval_batch_size=16
        fi
        
        # Create output directory
        output_dir="${BASE_OUTPUT_DIR}/${model_name}"
        mkdir -p "$output_dir"
        
        # Run 4 iterations of generate-then-train
        for iteration in {0..3}; do
            log_message "Starting iteration $iteration for $model_name"
            
            # First generate
            run_iteration "$model_name" "$model_path" "$dataset" "$iteration" "generate" \
                "$max_new_tokens" "$batch_size" "$grad_steps" "$output_dir"
                
            # Then train
            run_iteration "$model_name" "$model_path" "$dataset" "$iteration" "train" \
                "$max_new_tokens" "$batch_size" "$grad_steps" "$output_dir"
                
            log_message "Completed iteration $iteration for $model_name"
            sleep 5
        done

        # After all iterations complete, run evaluation on the checkpoint directory
        ckpts_dir="${output_dir}/mr/iteration_3/ckpts"
        if [ -d "$ckpts_dir" ]; then
            log_message "Running evaluation on checkpoints in: $ckpts_dir"
            ./src/scripts/run_ft_evaluation.sh "$ckpts_dir" "$dataset" "$eval_batch_size" | tee -a "$LOG_FILE"
        else
            log_message "Warning: Checkpoint directory not found: $ckpts_dir"
        fi
        
        log_message "Completed all iterations for $model_name"
    done
done

log_message "Completed all training combinations"