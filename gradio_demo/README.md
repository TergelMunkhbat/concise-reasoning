# Concise Reasoning Demo for Local Use

This directory contains a Gradio interface for running our models locally on your machine.

## Overview

This Gradio demo provides an easy-to-use web interface for interacting with our concise reasoning models. It allows you to quickly test and use any of our fine-tuned models on your local machine without requiring complex setup or deployment procedures.

## Available Models

We provide 10 fine-tuned models for concise reasoning:

- **LLaMA-3.2 Models**:
  - `tergel/llama-3.2-3b-instruct-gsm8k-fs-gpt4o-bon`
  - `tergel/llama-3.2-3b-instruct-math-fs-gpt4o-bon`

- **Qwen2.5 Models**:
  - `tergel/qwen2.5-3b-instruct-gsm8k-fs-gpt4o-bon`
  - `tergel/qwen2.5-3b-instruct-math-fs-gpt4o-bon`
  - `tergel/qwen2.5-math-1.5b-instruct-gsm8k-fs-gpt4o-bon`
  - `tergel/qwen2.5-math-1.5b-instruct-math-fs-gpt4o-bon`

- **Gemma-2 Models**:
  - `tergel/gemma-2-2b-it-gsm8k-fs-gpt4o-bon`
  - `tergel/gemma-2-2b-it-math-fs-gpt4o-bon`

- **DeepSeek Models**:
  - `tergel/deepseek-math-7b-instruct-gsm8k-fs-gpt4o-bon`
  - `tergel/deepseek-math-7b-instruct-math-fs-gpt4o-bon`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TergelMunkhbat/concise-reasoning.git
   cd concise-reasoning/gradio_demo
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Demo

To start the Gradio interface:

```bash
python app.py --model tergel/llama-3.2-3b-instruct-gsm8k-fs-gpt4o-bon
```

This will launch a local web server, typically at http://127.0.0.1:7860. Open this URL in your web browser to access the interface.

You can modify the `--model` argument to use any of our models listed above.

## Usage

1. Access the web interface at http://127.0.0.1:7860
2. Type your math problem or reasoning question in the chat input
3. Adjust generation parameters if needed (temperature, top-p, etc.)
4. View the model's concise reasoning solution

## License

This demo is distributed under the same license as the main project.
