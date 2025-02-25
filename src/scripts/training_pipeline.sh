#!/bin/bash

#########################################################
# LLM Training Pipeline
#
#
# USAGE:
#   ./training_pipeline.sh
#
# ENVIRONMENT VARIABLES:
#   BASE_MODEL_DIR    - Directory containing pretrained models
#   BASE_DATA_DIR     - Directory for dataset storage
#   BASE_FEW_SHOT_DIR - Directory containing few-shot examples
#   BASE_OUTPUT_DIR   - Directory where trained models will be saved
#   LOG_DIR           - Directory for log files
#   TEMP_DIR          - Directory for temporary files
#   CUDA_DEVICES      - Comma-separated list of GPU devices to use
#   TRAINING_TYPE     - "simple" or "augmented"
#########################################################


# Environment settings
CUDA_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# Base paths
BASE_MODEL_DIR="./models"
BASE_DATA_DIR="./data"
BASE_FEW_SHOT_DIR="./data/few_shot_examples"
BASE_OUTPUT_DIR="./models/trained"
LOG_DIR="logs"
TEMP_DIR="/tmp/hybrid_training"

# Add main process port for accelerate
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29501}

# Global variables to store data paths
GENERATED_DATA_PATH=""
ZERO_SHOT_DATA_PATH=""
FEW_SHOT_DATA_PATH=""

# Model configurations
# Format: "model_name:batch_size"
declare -a model_configs=(
    #"llama-3.2-1b-instruct:64"
    "llama-3.2-3b-instruct:32"
    #"qwen2.5-math-1.5b-instruct:32"
    #"qwen2.5-3b-instruct:32"
    #"gemma-2-2b-it:32"
    #"llama-3.1-8b-instruct:32"
    #"deepseek-math-7b-instruct:32"
)

# Dataset configurations
declare -a datasets=(
    #"gsm8k"
    "math"
)

# Training type - single option
# Options: "simple" or "augmented"
TRAINING_TYPE=${TRAINING_TYPE:-"augmented"} 

# Simple training type - choose one approach
# Options: "zero-shot" or "few-shot"
SIMPLE_APPROACH=${SIMPLE_APPROACH:-"zero-shot"} 

# Generation configurations - with diverse path counts
# Format: "method_type:number_of_diverse_path"
ZERO_SHOT_PROMPT_SYSTEM=${ZERO_SHOT_PROMPT_SYSTEM:-"irpo:16"}
FEW_SHOT_PROMPT_SYSTEM=${FEW_SHOT_PROMPT_SYSTEM:-"gpt4o:16"}

# Training configurations
# Used for trainer type (shortest or all)
USE_SHORTEST=true  

# Create necessary directories
mkdir -p "$TEMP_DIR"

# Log file
LOG_FILE="$LOG_DIR/pipeline.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Create log file
touch "$LOG_FILE"
log_message "Starting unified training pipeline"

# Create necessary log directories
mkdir -p "$LOG_DIR/generation"
mkdir -p "$LOG_DIR/training" 
mkdir -p "$LOG_DIR/evaluation"

# Function to get few-shot path
get_few_shot_path() {
    local model_name=$1
    local dataset=$2
    local prompt_system_key=$3
    
    if [[ "$prompt_system_key" == "gpt4o" || "$prompt_system_key" == "original-cot" ]]; then
        echo "${BASE_FEW_SHOT_DIR}/${dataset}/few-shot-${prompt_system_key}.json"
    else
        echo "${BASE_FEW_SHOT_DIR}/filtered_few_shot_prompts/${dataset}/${model_name}/${prompt_system_key}_128_128.json"
    fi
}

# Function to set up and run generation
run_generation() {
    local model_name=$1
    local batch_size=$2
    local dataset=$3
    local prompt_type=$4
    local prompt_system=$5
    local num_diverse_path=$6
    
    log_message "Running generation for $model_name with $prompt_type prompt system: $prompt_system (diverse paths: $num_diverse_path)"
    
    # Set max_new_tokens based on dataset
    local max_new_tokens=512
    if [ "$dataset" = "math" ]; then
        max_new_tokens=1024
    fi
    
    # Define the generation path (using global variable)
    GENERATED_DATA_PATH="${BASE_DATA_DIR}/${dataset}/${model_name}/${prompt_type}/${prompt_system}/${max_new_tokens}/0.7/40/0.95/${num_diverse_path}/${batch_size}/"
    log_message "Generated data path: $GENERATED_DATA_PATH"
    
    local cmd="python src/reasoning_generation.py \
        --prompt $prompt_type \
        --model_path \"${BASE_MODEL_DIR}/${model_name}\" \
        --model_name \"$model_name\" \
        --dataset \"$dataset\" \
        --do_sample \
        --batch_size \"$batch_size\" \
        --num_diverse_path $num_diverse_path \
        --max_new_tokens $max_new_tokens \
        --use_vllm"

    if [ "$prompt_type" = "zero-shot" ]; then
        cmd="$cmd --prompt_system $prompt_system"
    elif [ "$prompt_type" = "few-shot" ]; then
        local few_shot_path=$(get_few_shot_path "$model_name" "$dataset" "$prompt_system")
        if [ ! -f "$few_shot_path" ]; then
            log_message "Warning: Few-shot path does not exist: $few_shot_path"
            return 1
        fi
        cmd="$cmd --few_shot_path \"$few_shot_path\" --prompt_system_key \"$prompt_system\" --prompt_system no"
    fi
    
    # Check if data already exists
    if ls "${GENERATED_DATA_PATH}"/*.json 1> /dev/null 2>&1; then
        log_message "Found existing data in ${GENERATED_DATA_PATH}, skipping generation"
        return 0
    else
        
        # Create log file directory if it doesn't exist
        mkdir -p "$(dirname "$LOG_DIR/generation/${prompt_type}_${prompt_system}_${dataset}_${model_name}.txt")"
        
        # Run the command and tee output to both terminal and log file
        eval "$cmd" 2>&1 | tee "$LOG_DIR/generation/${prompt_type}_${prompt_system}_${dataset}_${model_name}.txt"
        generation_result=$?
        
        if [ $generation_result -ne 0 ]; then
            log_message "Generation failed with exit code: $generation_result"
            return $generation_result
        fi
        
        log_message "Generation completed successfully"
        return 0
    fi
}

# Function to run training
run_training() {
    local model_name=$1
    local dataset=$2
    local output_dir=$3
    local data_path=$4
    
    # Set batch size and gradient steps based on dataset
    local batch_size=16
    local grad_steps=1
    if [ "$dataset" = "math" ]; then
        batch_size=8
        grad_steps=2
    fi

    # Set trainer type based on USE_SHORTEST
    local trainer_type="all"
    if [ "$USE_SHORTEST" = true ]; then
        trainer_type="shortest"
    fi

    mkdir -p "$output_dir"

    # Execute training command with improved logging
    log_message "Running training command for $model_name on $dataset using data from $data_path"
    
    mkdir -p "$(dirname "$LOG_DIR/training/$(basename "$output_dir").txt")"
    
    python src/training.py --config-name=sft_train \
        trainer.model_name=$model_name \
        trainer.model_path="${BASE_MODEL_DIR}/${model_name}" \
        trainer.dataset=$dataset \
        trainer.data_path=$data_path \
        trainer.type=$trainer_type \
        trainer.output_dir=$output_dir \
        trainer.batch_size=$batch_size \
        trainer.grad_steps=$grad_steps \
        trainer.use_raw_output_dir=True \
        2>&1 | tee "$LOG_DIR/training/$(basename "$output_dir")_${dataset}.txt"
    
    training_result=$?
    log_message "Training completed with exit code: $training_result"
    return $training_result
}

# Function to run evaluation
run_evaluation() {
    local ckpt_dir=$1
    local dataset=$2
    local batch_size=$3
    local model_name=$4
    
    # Set max_new_tokens based on dataset
    local max_new_tokens=512
    if [ "$dataset" = "math" ]; then
        max_new_tokens=1024
    fi
    
    log_message "Running evaluation on $ckpt_dir for dataset $dataset"
    
    mkdir -p "$(dirname "$LOG_DIR/evaluation/$(basename "$(dirname "$ckpt_dir")").txt")"
    
    # Run with vLLM
    python src/evaluation.py \
        --model_path "$ckpt_dir" \
        --model_name "$model_name" \
        --dataset "$dataset" \
        --prompt direct \
        --prompt_system no \
        --max_new_tokens "$max_new_tokens" \
        --batch_size "$batch_size" \
        --use_vllm 2>&1 | tee "$LOG_DIR/evaluation/$(basename "$(dirname "$ckpt_dir")")_${dataset}.txt"
    
    eval_result=$?
    log_message "Evaluation completed with exit code: $eval_result"
    return $eval_result
}

# Main execution loop
for dataset in "${datasets[@]}"; do
    for model_config in "${model_configs[@]}"; do
        IFS=':' read -r model_name batch_size <<< "$model_config"
        
        log_message "Starting $TRAINING_TYPE training pipeline for $model_name on $dataset"
        
        # Generate data based on training type and approach
        if [ "$TRAINING_TYPE" = "simple" ]; then
            if [ "$SIMPLE_APPROACH" = "zero-shot" ]; then
                # Split zero-shot prompt system info
                IFS=':' read -r zero_shot_system zero_shot_diverse_path <<< "$ZERO_SHOT_PROMPT_SYSTEM"
                
                # Generate zero-shot data only
                run_generation "$model_name" "$batch_size" "$dataset" "zero-shot" "$zero_shot_system" "$zero_shot_diverse_path"
                generation_exit_code=$?
                if [ $generation_exit_code -ne 0 ]; then
                    log_message "Error in zero-shot generation phase"
                    continue
                fi
                
                # Store zero-shot data path
                ZERO_SHOT_DATA_PATH="$GENERATED_DATA_PATH"
                
                # Check if preprocessing is needed for zero-shot data
                if [ -d "${ZERO_SHOT_DATA_PATH}/train" ]; then
                    log_message "Preprocessed zero-shot data already exists in ${ZERO_SHOT_DATA_PATH}, skipping preprocessing"
                else
                    # Preprocess zero-shot data
                    log_message "Preprocessing zero-shot data at: $ZERO_SHOT_DATA_PATH"
                    python src/preprocess.py --data_dir "$ZERO_SHOT_DATA_PATH" --dataset "$dataset"
                    if [ $? -ne 0 ]; then
                        log_message "Error in zero-shot preprocessing phase"
                        continue
                    fi
                fi
            elif [ "$SIMPLE_APPROACH" = "few-shot" ]; then
                # Split few-shot prompt system info
                IFS=':' read -r few_shot_system few_shot_diverse_path <<< "$FEW_SHOT_PROMPT_SYSTEM"
                
                # Generate few-shot data only
                run_generation "$model_name" "$batch_size" "$dataset" "few-shot" "$few_shot_system" "$few_shot_diverse_path"
                if [ $? -ne 0 ]; then
                    log_message "Error in few-shot generation phase"
                    continue
                fi
                
                # Store few-shot data path
                FEW_SHOT_DATA_PATH="$GENERATED_DATA_PATH"
                
                # Check if preprocessing is needed for few-shot data
                if [ -d "${FEW_SHOT_DATA_PATH}/train" ]; then
                    log_message "Preprocessed few-shot data already exists in ${FEW_SHOT_DATA_PATH}, skipping preprocessing"
                else
                    # Preprocess few-shot data
                    log_message "Preprocessing few-shot data at: $FEW_SHOT_DATA_PATH"
                    python src/preprocess.py --data_dir "$FEW_SHOT_DATA_PATH" --dataset "$dataset"
                    if [ $? -ne 0 ]; then
                        log_message "Error in few-shot preprocessing phase"
                        continue
                    fi
                fi
            else
                log_message "Invalid SIMPLE_APPROACH: $SIMPLE_APPROACH. Must be 'zero-shot' or 'few-shot'."
                continue
            fi
        elif [ "$TRAINING_TYPE" = "augmented" ]; then
            # For augmented training, generate both zero-shot and few-shot data
            # Split prompt system info
            IFS=':' read -r zero_shot_system zero_shot_diverse_path <<< "$ZERO_SHOT_PROMPT_SYSTEM"
            IFS=':' read -r few_shot_system few_shot_diverse_path <<< "$FEW_SHOT_PROMPT_SYSTEM"
            
            # Generate zero-shot data first
            run_generation "$model_name" "$batch_size" "$dataset" "zero-shot" "$zero_shot_system" "$zero_shot_diverse_path"
            generation_exit_code=$?
            if [ $generation_exit_code -ne 0 ]; then
                log_message "Error in zero-shot generation phase"
                continue
            fi
            
            # Store zero-shot data path
            ZERO_SHOT_DATA_PATH="$GENERATED_DATA_PATH"
            
            # Check if preprocessing is needed for zero-shot data
            if [ -d "${ZERO_SHOT_DATA_PATH}/train" ]; then
                log_message "Preprocessed zero-shot data already exists in ${ZERO_SHOT_DATA_PATH}, skipping preprocessing"
            else
                # Preprocess zero-shot data
                log_message "Preprocessing zero-shot data at: $ZERO_SHOT_DATA_PATH"
                python src/preprocess.py --data_dir "$ZERO_SHOT_DATA_PATH" --dataset "$dataset"
                if [ $? -ne 0 ]; then
                    log_message "Error in zero-shot preprocessing phase"
                    continue
                fi
            fi
            
            # Generate few-shot data
            run_generation "$model_name" "$batch_size" "$dataset" "few-shot" "$few_shot_system" "$few_shot_diverse_path"
            if [ $? -ne 0 ]; then
                log_message "Error in few-shot generation phase"
                continue
            fi
            
            # Store few-shot data path
            FEW_SHOT_DATA_PATH="$GENERATED_DATA_PATH"
            
            # Check if preprocessing is needed for few-shot data
            if [ -d "${FEW_SHOT_DATA_PATH}/train" ]; then
                log_message "Preprocessed few-shot data already exists in ${FEW_SHOT_DATA_PATH}, skipping preprocessing"
            else
                # Preprocess few-shot data
                log_message "Preprocessing few-shot data at: $FEW_SHOT_DATA_PATH"
                python src/preprocess.py --data_dir "$FEW_SHOT_DATA_PATH" --dataset "$dataset"
                if [ $? -ne 0 ]; then
                    log_message "Error in few-shot preprocessing phase"
                    continue
                fi
            fi
        fi
        
        # Run training and evaluation based on training type
        if [ "$TRAINING_TYPE" = "simple" ]; then
            # Simple training pipeline - train with either zero-shot or few-shot data based on SIMPLE_APPROACH
            if [ "$SIMPLE_APPROACH" = "zero-shot" ]; then
                log_message "Using simple training pipeline with zero-shot data"
                # Split zero-shot prompt system info
                IFS=':' read -r zero_shot_system zero_shot_diverse_path <<< "$ZERO_SHOT_PROMPT_SYSTEM"
                simple_output_dir="${BASE_OUTPUT_DIR}/simple/${model_name}/${dataset}/zero_${zero_shot_system}_${zero_shot_diverse_path}"
                
                # Check if training has already been completed
                if [ -d "${simple_output_dir}/ckpts" ]; then
                    log_message "Training already completed, checkpoint exists at ${simple_output_dir}/ckpts"
                else
                    # Train with zero-shot data
                    run_training "$model_name" "$dataset" "$simple_output_dir" "$ZERO_SHOT_DATA_PATH"
                    if [ $? -ne 0 ]; then
                        log_message "Error in training phase"
                        continue
                    fi
                fi
                
                # Evaluate model
                run_evaluation "${simple_output_dir}/ckpts/" "$dataset" "$batch_size" "$model_name"
                
            elif [ "$SIMPLE_APPROACH" = "few-shot" ]; then
                log_message "Using simple training pipeline with few-shot data"
                # Split few-shot prompt system info
                IFS=':' read -r few_shot_system few_shot_diverse_path <<< "$FEW_SHOT_PROMPT_SYSTEM"
                simple_output_dir="${BASE_OUTPUT_DIR}/simple/${model_name}/${dataset}/few_${few_shot_system}_${few_shot_diverse_path}"
                
                # Check if training has already been completed
                if [ -d "${simple_output_dir}/ckpts" ]; then
                    log_message "Training already completed, checkpoint exists at ${simple_output_dir}/ckpts"
                else
                    # Train with few-shot data
                    run_training "$model_name" "$dataset" "$simple_output_dir" "$FEW_SHOT_DATA_PATH"
                    if [ $? -ne 0 ]; then
                        log_message "Error in training phase"
                        continue
                    fi
                fi
                
                # Evaluate model
                run_evaluation "${simple_output_dir}/ckpts/" "$dataset" "$batch_size" "$model_name"
            else
                log_message "Invalid SIMPLE_APPROACH: $SIMPLE_APPROACH. Must be 'zero-shot' or 'few-shot'."
                continue
            fi
        elif [ "$TRAINING_TYPE" = "augmented" ]; then
            # Augmented training pipeline - combine zero-shot and few-shot data
            log_message "Using augmented training pipeline - combining zero-shot and few-shot data"
            
            # Extract prompt systems and diverse paths
            IFS=':' read -r zero_shot_system zero_shot_diverse_path <<< "$ZERO_SHOT_PROMPT_SYSTEM"
            IFS=':' read -r few_shot_system few_shot_diverse_path <<< "$FEW_SHOT_PROMPT_SYSTEM"
            
            # Create temporary directory for combined data
            temp_dir="${TEMP_DIR}/${model_name}_${dataset}_${zero_shot_system}_${few_shot_system}"
            rm -rf "$temp_dir"
            mkdir -p "${temp_dir}/train"
            mkdir -p "${temp_dir}/valid"
            
            # Combine the data
            log_message "Combining data from $ZERO_SHOT_DATA_PATH and $FEW_SHOT_DATA_PATH"
            
            # Copy zero-shot data with prefix
            if [ -d "${ZERO_SHOT_DATA_PATH}/train" ]; then
                for file in "${ZERO_SHOT_DATA_PATH}/train"/*.json; do
                    if [ -f "$file" ]; then
                        cp "$file" "${temp_dir}/train/zero_$(basename "$file")"
                    fi
                done
            fi
            
            # Copy few-shot data with prefix
            if [ -d "${FEW_SHOT_DATA_PATH}/train" ]; then
                for file in "${FEW_SHOT_DATA_PATH}/train"/*.json; do
                    if [ -f "$file" ]; then
                        cp "$file" "${temp_dir}/train/few_$(basename "$file")"
                    fi
                done
            fi
            
            # Copy validation data (from zero-shot)
            if [ -d "${ZERO_SHOT_DATA_PATH}/valid" ]; then
                cp "${ZERO_SHOT_DATA_PATH}/valid"/*.json "${temp_dir}/valid/" 2>/dev/null || true
            fi
            
            # Define output directory for augmented training
            augmented_output_dir="${BASE_OUTPUT_DIR}/augmented/${model_name}/${dataset}/${zero_shot_system}_${zero_shot_diverse_path}_${few_shot_system}_${few_shot_diverse_path}"
            
            # Check if augmented training has already been completed
            if [ -d "${augmented_output_dir}/ckpts" ]; then
                log_message "Augmented training already completed, checkpoint exists at ${augmented_output_dir}/ckpts"
            else
                # Train with combined data
                run_training "$model_name" "$dataset" "$augmented_output_dir" "$temp_dir"
                if [ $? -ne 0 ]; then
                    log_message "Error in augmented training phase"
                    rm -rf "$temp_dir"
                    continue
                fi
            fi
            
            # Evaluate augmented model
            run_evaluation "${augmented_output_dir}/ckpts/" "$dataset" "$batch_size" "$model_name"
            
            # Clean up
            rm -rf "$temp_dir"
        fi
        
        log_message "Completed pipeline for $model_name on $dataset with $TRAINING_TYPE training"
        sleep 3
    done
done

log_message "Completed all training pipelines"