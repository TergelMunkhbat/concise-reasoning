import os
import argparse
from collections.abc import Iterator
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Token limits
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 512

# Description
DESCRIPTION = """\
# Demo for "Self-Training Elicits Concise Reasoning in Large Language Models"

This is a simple chat interface allowing you to observe the concise Chain-of-Thought (CoT) solutions that our model can produce.
Feel free to try different models.
"""

def create_demo(model_id="tergel/llama-3.2-3b-instruct-gsm8k-fs-gpt4o-bon"):
    """Create and return the Gradio demo for the specified model."""
    # Decide on device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    description = DESCRIPTION
    if not torch.cuda.is_available():
        description += "\n\n<p>**Warning**: Running on CPU 🥶 – this may be extremely slow.</p>"
        
    print(f"Loading model: {model_id}")
    description += f"\n\n<p>Model: {model_id}</p>"
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=None if device == "cpu" else "auto",
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False

    def generate(
        message: str,
        chat_history: list[dict],
        system_prompt: str = "",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.2,
    ) -> Iterator[str]:
        # Build conversation
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        conversation += chat_history
        conversation.append({"role": "user", "content": message})
        
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True)
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
        input_ids = input_ids.to(model.device)
        
        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
            repetition_penalty=repetition_penalty,
        )
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        
        outputs = []
        for text in streamer:
            outputs.append(text)
            yield "".join(outputs)
        
    chat_interface = gr.ChatInterface(
        fn=generate,
        additional_inputs=[
            gr.Textbox(label="System prompt", lines=6),
            gr.Slider(
                label="Max new tokens",
                minimum=1,
                maximum=MAX_MAX_NEW_TOKENS,
                step=1,
                value=DEFAULT_MAX_NEW_TOKENS,
            ),
            gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=4.0,
                step=0.1,
                value=0.7,
            ),
            gr.Slider(
                label="Top-p (nucleus sampling)",
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                value=0.95,
            ),
            gr.Slider(
                label="Top-k",
                minimum=1,
                maximum=1000,
                step=1,
                value=40,
            ),
            gr.Slider(
                label="Repetition penalty",
                minimum=1.0,
                maximum=2.0,
                step=0.05,
                value=1.2,
            ),
        ],
        stop_btn=None,
        examples=[
            [
                "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"
            ],
            [
                "Claire makes a 3 egg omelet every morning for breakfast. How many dozens of eggs will she eat in 4 weeks?"
            ],
            [
                "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?"
            ],
        ],
        cache_examples=False,
        type="messages",
    )

    with gr.Blocks(fill_height=True) as demo:
        gr.Markdown(description)          
        chat_interface.render()
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Gradio demo for concise reasoning models")
    parser.add_argument(
        "--model", 
        type=str, 
        default="tergel/llama-3.2-3b-instruct-gsm8k-fs-gpt4o-bon",
        help="Model ID to use for the demo"
    )
    args = parser.parse_args()
    
    demo = create_demo(model_id=args.model)
    demo.queue(max_size=20).launch()