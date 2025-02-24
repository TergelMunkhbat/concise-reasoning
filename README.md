# Self-Training Elicits Concise Reasoning in Large Language Models

Official repository for the paper "Self-Training Elicits Concise Reasoning in Large Language Models," by Tergel Munkhbat, Namgyu Ho, Seo Hyun Kim, Yongjin Yang, Yujin Kim, and Se-Young Yun.

## Environment Setup

1. Setup Conda environment:
```bash
conda create --name concise python=3.12 -y
conda activate concise
pip install -r requirements.txt
```

## Directory Structure

The pipeline expects the following directory structure:
```
.
├── models/                    # Pre-trained models
│   ├── llama-3.2-1b-instruct/ 
│   ├── llama-3.2-3b-instruct/
│   └── ...
├── data/                      # Dataset and generated data
│   ├── gsm8k/
│   ├── math/
│   └── few_shot_examples/     # Few-shot examples for prompting
```

## Supported Models and Datasets

### Models
- Llama 3.2 (1B and 3B)
- Qwen 2.5 (Math 1.5B and 3B)
- Gemma 2 (2B)
- Llama 3.1 (8B)
- DeepSeek Math (7B)

### Datasets
- GSM8K: Grade-school math word problems
- MATH: Competition-level mathematics problems

## Training Pipeline

Our training pipeline (`training_pipeline.sh`) supports two primary training modes:

1. **Simple Training**: Train using either zero-shot or few-shot generated data
2. **Augmented Training**: Train using a combination of both approaches

## Reproducing Paper Results

1. Clone the repository
2. Download the pre-trained models and place them in the `models/` directory:
3. Run the training pipeline:

```bash
TRAINING_TYPE="augmented" ZERO_SHOT_PROMPT_SYSTEM="irpo:16" FEW_SHOT_PROMPT_SYSTEM="gpt4o:16" ./training_pipeline.sh
```

This command will:
- Generate 16 diverse reasoning paths for each problem using both IRPO zero-shot and GPT-4o few-shot approaches
- Combine these datasets for augmented training
- Train models with the shortest correct reasoning path for each question
- Evaluate models on test sets and report accuracy metrics

## Understanding the Training Pipeline

The `training_pipeline.sh` script orchestrates the entire training process:

1. **Generation Phase**: Creates reasoning paths using specified prompting approaches

2. **Preprocessing Phase**: Converts generated paths into a training-ready format

3. **Training Phase**: Fine-tunes the model on the generated data

4. **Evaluation Phase**: Tests model performance on benchmark datasets

## Customization Options

You can modify the following parameters in the script:

- `TRAINING_TYPE`: Choose "simple" or "augmented"
- `SIMPLE_APPROACH`: If using simple training, choose "zero-shot" or "few-shot"
- `ZERO_SHOT_PROMPT_SYSTEM`: Format is "method:num_paths" (e.g., "irpo:16")
- `FEW_SHOT_PROMPT_SYSTEM`: Format is "method:num_paths" (e.g., "gpt4o:16")
- `USE_SHORTEST`: Set to true to use only the shortest rationales during training
- `CUDA_DEVICES`: Specify which GPUs to use (e.g., "0,1,2,3")

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{munkhbat2024selftraining,
  title={Self-Training Elicits Concise Reasoning in Large Language Models},
  author={Munkhbat, Tergel and Ho, Namgyu and Kim, Seo Hyun and Yang, Yongjin and Kim, Yujin and Yun, Se-Young},
  journal={},
  year={2025}
}
```

## License

[MIT License](LICENSE)

