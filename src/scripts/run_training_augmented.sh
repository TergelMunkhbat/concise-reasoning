#!/bin/bash

# Environment settings
CUDA_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# Base paths
BASE_MODEL_DIR="./models"
BASE_DATA_DIR="./data"
BASE_OUTPUT_DIR="./models/trained/augmented"
LOG_DIR="logs"
TEMP_DIR="/tmp/hybrid_training"

# Model configurations
# Format: "model_name:eval_batch_size"
declare -a model_configs=(
    "llama-3.2-1b-instruct:64"
    "llama-3.2-3b-instruct:32"
    "qwen2.5-math-1.5b-instruct:32"
    "qwen2.5-3b-instruct:32"
    "gemma-2-2b-it:32"
    "llama-3.1-8b-instruct:32"
    "deepseek-math-7b-instruct:32"
)

# Training configurations for augmented training
declare -a base_types=(
    "ft_16"
)

# Define if using shortest rationales
USE_SHORTEST=true

# Training configurations for augmented training
declare -a extra_types=(
    "fs_gpt4o_16"
)

# Dataset configurations
declare -a datasets=(
    "gsm8k"
    "math"
)

# Create necessary directories
mkdir -p "$LOG_DIR/training"
mkdir -p "$LOG_DIR/evaluation"
mkdir -p "$TEMP_DIR"

# Log file
LOG_FILE="$LOG_DIR/training_evaluation.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to get the config suffix based on shift setting
get_config_suffix() {
    if [ "$USE_SHORTEST" = true ]; then
        echo "_shift"
    else
        echo ""
    fi
}

# Function to combine training data
combine_training_data() {
    local model_name=$1
    local dataset=$2
    local base_type=$3
    local extra_type=$4
    local temp_output_dir=$5

    # Create temporary directories for combined data
    mkdir -p "${temp_output_dir}/train"

    # Base paths
    local base_path="${BASE_DATA_DIR}/${dataset}/${base_type}/${model_name}"
    local extra_path="${BASE_DATA_DIR}/${dataset}/${extra_type}/${model_name}"

    # Combine training data with prefixes to avoid name conflicts
    if [ -d "${base_path}/train" ] && [ -d "${extra_path}/train" ]; then
        # Copy base files with base_ prefix
        for file in "${base_path}/train"/*; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                cp "$file" "${temp_output_dir}/train/base_${filename}"
            fi
        done
        
        # Copy extra files with extra_ prefix
        for file in "${extra_path}/train"/*; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                cp "$file" "${temp_output_dir}/train/extra_${filename}"
            fi
        done
    fi

    echo "${temp_output_dir}"
}

# Create log file
touch "$LOG_FILE"
log_message "Starting hybrid training and evaluation automation"

# Calculate total combinations (only hybrid combinations)
total_combinations=$((${#model_configs[@]} * ${#base_types[@]} * ${#extra_types[@]} * ${#datasets[@]}))
current=0

# Main training loop
for DATASET in "${datasets[@]}"; do
    for model_config in "${model_configs[@]}"; do
        # Split model configuration
        IFS=':' read -r model_name eval_batch_size <<< "$model_config"
        model_path="${BASE_MODEL_DIR}/${model_name}"

        for base_type in "${base_types[@]}"; do
            for extra_type in "${extra_types[@]}"; do
                ((current++))

                # Get configuration suffix for paths and logs
                config_suffix=$(get_config_suffix)
                current_base_type="${base_type}${config_suffix}"

                log_message "Starting hybrid training $current/$total_combinations"
                log_message "Dataset: $DATASET, Model: $model_name, Base: $current_base_type, Extra: $extra_type"

                # Create temporary directory for combined data
                temp_dir="${TEMP_DIR}/${model_name}_${DATASET}_${current_base_type}_${extra_type}"
                rm -rf "$temp_dir"  # Clean up any previous data
                combined_data_path=$(combine_training_data "$model_name" "$DATASET" "$base_type" "$extra_type" "$temp_dir")

                output_dir="${BASE_OUTPUT_DIR}/${model_name}/${DATASET}/${current_base_type}_${extra_type}"
                mkdir -p "$output_dir"

                # Set batch size and gradient steps based on dataset
                if [ "$DATASET" = "math" ]; then
                    batch_size=8
                    grad_steps=2
                else
                    batch_size=16
                    grad_steps=1
                fi

                # Set training parameters based on shift configuration
                if [ "$USE_SHORTEST" = true ]; then
                    trainer_type="shortest"
                else
                    trainer_type="all"
                fi

                # Run training with combined data
                python src/training.py --config-name=sft_train \
                    trainer.model_name=$model_name \
                    trainer.model_path=$model_path \
                    trainer.dataset=$DATASET \
                    trainer.data_path=$combined_data_path \
                    trainer.type=$trainer_type \
                    trainer.output_dir=$output_dir \
                    trainer.batch_size=$batch_size \
                    trainer.grad_steps=$grad_steps \
                    trainer.use_raw_output_dir=True \
                    2>&1 | tee "$LOG_DIR/training/${model_name}_${DATASET}_${current_base_type}_${extra_type}.txt"

                # Check if training was successful
                if [ $? -eq 0 ]; then
                    log_message "Successfully completed training for hybrid ${current_base_type}_${extra_type}"

                    # Run evaluation
                    log_message "Starting evaluation"
                    src/scripts/run_ft_evaluation.sh \
                        "$output_dir/ckpts/" \
                        "$DATASET" \
                        "$eval_batch_size" \
                        2>&1 | tee "$LOG_DIR/evaluation/${model_name}_${DATASET}_${current_base_type}_${extra_type}.txt"

                    if [ $? -eq 0 ]; then
                        log_message "Successfully completed evaluation"
                    else
                        log_message "Error in evaluation"
                    fi
                else
                    log_message "Error in training combination $current/$total_combinations"
                fi

                # Clean up temporary directory
                rm -rf "$temp_dir"
                sleep 3
            done # end extra_type loop
        done # end base_type loop
    done # end model_config loop
done # end DATASET loop

log_message "Completed all training combinations"