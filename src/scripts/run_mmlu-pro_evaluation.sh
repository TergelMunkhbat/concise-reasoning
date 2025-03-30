#!/bin/bash

# OOD Evaluation Script for Trained Models
# This script runs evaluations using the MMLU-Pro dataset for models trained on either GSM8K or MATH datasets

# Set default batch size
DEFAULT_BATCH_SIZE=64

# Check if batch size is provided as argument
if [ $# -eq 1 ]; then
    BATCH_SIZE=$1
else
    BATCH_SIZE=$DEFAULT_BATCH_SIZE
fi

echo "Starting MMLU evaluation with batch size: $BATCH_SIZE"
echo "=============================================="

# Function to evaluate a model
evaluate_model() {
    local model_path=$1
    local train_dataset=$2  # The dataset the model was trained on
    local eval_dataset="mmlu-pro"   # Using MMLU as the evaluation dataset
    
    echo "Evaluating model: $model_path"
    echo "Training dataset: $train_dataset, Evaluation dataset: $eval_dataset"
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Create a log filename based on the model name and evaluation dataset
    log_filename="logs/$(basename "$model_path")_${eval_dataset}_eval.log"
    
    echo "Running evaluation and writing output to $log_filename"
    
    # Run the evaluation script and redirect output to log file
    ./src/scripts/run_ft_evaluation.sh "$model_path" "$eval_dataset" "$BATCH_SIZE" >> "$log_filename" 2>&1
    
    echo "Completed evaluation for $model_path on $eval_dataset"
    echo "--------------------------------------------"
}

# Define models and their training datasets
declare -A models
# LLaMA-3.2 Models
models["tergel/llama-3.2-3b-instruct-gsm8k-fs-gpt4o-bon"]="gsm8k"
models["tergel/llama-3.2-3b-instruct-math-fs-gpt4o-bon"]="math"

# Qwen2.5 Models
models["tergel/qwen2.5-3b-instruct-gsm8k-fs-gpt4o-bon"]="gsm8k"
models["tergel/qwen2.5-3b-instruct-math-fs-gpt4o-bon"]="math"

# Gemma-2 Models
models["tergel/gemma-2-2b-it-gsm8k-fs-gpt4o-bon"]="gsm8k"
models["tergel/gemma-2-2b-it-math-fs-gpt4o-bon"]="math"

# Create results directory if it doesn't exist
mkdir -p mmlu_evaluation_results

# Start evaluation for each model
for model_path in "${!models[@]}"; do
    train_dataset=${models[$model_path]}
    
    # Evaluate the model
    evaluate_model "$model_path" "$train_dataset"
done

echo "All evaluations complete!"
echo "Results should be available in your output directory."

# Optional: Generate a summary report
echo "Generating summary report..."
echo "MMLU Evaluation Summary" > mmlu_evaluation_summary.txt
echo "======================" >> mmlu_evaluation_summary.txt
echo "Date: $(date)" >> mmlu_evaluation_summary.txt
echo "" >> mmlu_evaluation_summary.txt

echo "Model, Trained On, Evaluated On, Accuracy" >> mmlu_evaluation_summary.txt
echo "Summary report generated: mmlu_evaluation_summary.txt"