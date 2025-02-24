#!/bin/bash

# Environment settings
CUDA_DEVICES="4,5,6,7"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# Base paths
BASE_MODEL_DIR="./models"
BASE_DATA_DIR="./data"
BASE_OUTPUT_DIR="./models/trained/augmented/"
LOG_DIR="logs"

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

# Training configurations
# Format: "training_type"
declare -a training_configs=(
    ft_16
)

# Define if using shortest rationales
USE_SHORTEST=true

# Dataset configurations
declare -a datasets=("gsm8k" "math")

# Create log directories
mkdir -p "$LOG_DIR/training"
mkdir -p "$LOG_DIR/evaluation"

# Log file
LOG_FILE="$LOG_DIR/training_evaluation.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Create log file
touch "$LOG_FILE"
log_message "Starting training and evaluation automation"

# Calculate total combinations
total_combinations=$((${#model_configs[@]} * ${#training_configs[@]} * ${#datasets[@]}))
current=0

# Loop through each dataset
for DATASET in "${datasets[@]}"; do
    # Loop through each model
    for model_config in "${model_configs[@]}"; do
        # Split model configuration
        IFS=':' read -r model_name eval_batch_size <<< "$model_config"
        
        # Construct model path
        model_path="${BASE_MODEL_DIR}/${model_name}"
        
        # Loop through each training configuration
        for training_config in "${training_configs[@]}"; do
            # Split training configuration
            IFS=':' read -r training_type <<< "$training_config"
            
            ((current++))
            log_message "Starting combination $current/$total_combinations"
            log_message "Dataset: $DATASET, Model: $model_name, Training Type: $training_type"
            
            # Construct data path based on training type
            if [[ "$training_type" == "no_cot" || "$training_type" == "manual_cot" ]]; then
                data_path="${BASE_DATA_DIR}/${DATASET}/${training_type}"
            else
                data_path="${BASE_DATA_DIR}/${DATASET}/${training_type}/${model_name}"
            fi

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

            output_dir="${BASE_OUTPUT_DIR}/${model_name}/${DATASET}/${training_type}_${trainer_type}"
            mkdir -p "$output_dir"
            
            # Run training
            python src/training.py --config-name=sft_train \
                trainer.model_name=$model_name \
                trainer.model_path=$model_path \
                trainer.dataset=$DATASET \
                trainer.data_path=$data_path \
                trainer.type=$trainer_type \
                trainer.output_dir=$output_dir \
                trainer.batch_size=$batch_size \
                trainer.grad_steps=$grad_steps \
                trainer.use_raw_output_dir=True \
                2>&1 | tee "$LOG_DIR/training/${model_name}_${DATASET}_${training_type}.txt"
                
            # Check if training was successful
            if [ $? -eq 0 ]; then
                log_message "Successfully completed training for $model_name with $training_type on $DATASET"

                # Evaluation phase
                log_message "Starting evaluation for $model_name with $training_type on $DATASET"
                src/scripts/run_ft_evaluation.sh \
                    "$output_dir/ckpts/" \
                    "$DATASET" \
                    "$eval_batch_size" \
                    2>&1 | tee "$LOG_DIR/evaluation/${model_name}_${DATASET}_${training_type}.txt"
                
                if [ $? -eq 0 ]; then
                    log_message "Successfully completed evaluation for $model_name with $training_type on $DATASET"
                else
                    log_message "Error in evaluation for $model_name with $training_type on $DATASET"
                fi
            else
                log_message "Error in training combination $current/$total_combinations"
            fi
            
            # Optional: Add a delay between training runs
            sleep 3
        done
    done
done

log_message "Completed all training combinations"