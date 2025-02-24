import os
import torch
import transformers
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple

IGNORE_INDEX = -100


def get_latest_checkpoint(model_path) -> str:
    """
    Get the latest checkpoint from an output directory. Useful when retrieving last checkpoint saved by HF Trainer.

    Parameters:
        model_path (str): model_path (may be an output directory containing multiple checkpoints).

    Returns:
        str: model_path or path to the latest checkpoint if applicable.
    """
    if os.path.exists(model_path):
        checkpoints = [f for f in os.listdir(model_path) if f.startswith("checkpoint-")]
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            print(f"Loading model from checkpoint: {checkpoints[-1]}")
            model_path = os.path.join(model_path, checkpoints[-1])
    return model_path

def load_model(model_path, device_map, type="test"):
    """
    Load the pretrained model and tokenizer from the specified path.

    Parameters:
        model_path (str): Path to the pretrained model.
        device_map (str): Device mapping for the model.
        type(str): Type of loading the model: test or train

    Returns:
        tuple: Loaded model and tokenizer
    """
    model_path = get_latest_checkpoint(model_path)
    
    # Update the config for better memory efficiency
    use_cache = False if type == "train" else True

    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map=device_map, 
        torch_dtype=torch.bfloat16,
        use_cache = use_cache
    )

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def load_model_with_flash_attention(model_path, device_map, mode="test") -> Tuple[torch.nn.Module, object]:
    """
    Load a model with Flash Attention enabled.
    
    Args:
        model_path: Path to the model or model identifier
        model_type: Type of model to load ("auto" or specific architecture)
        mode: Whether to load for "train" or "inference"
    
    Returns:
        tuple: (model, tokenizer)
    """
    model_path = get_latest_checkpoint(model_path)

    # Update the config for better memory efficiency
    use_cache = False if mode == "train" else True

    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)

    # Load model with Flash Attention configuration
    attn_implementation = "flash_attention_2"
    if "gemma" in model_path:
        attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        use_cache = use_cache
    )

    # Set pad token if not present
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    DataCollator for supervised datasets, handling padding and conversion to tensors
    for input_ids and labels. This class ensures that the input data is properly
    formatted for use in model training.
    
    Attributes:
    -----------
    tokenizer: transformers.PreTrainedModel
        A tokenizer that will be used to convert input data into token IDs and provide 
        special tokens such as the padding token.
    """

    tokenizer: transformers.PreTrainedModel

    def __call__(self, instances: list) -> dict:
        """
        Process a list of instances and return a dictionary containing padded input IDs, labels, and attention masks

        Parameters:
        -----------
        instances: list
            A list of dictionaries where each dictionary contains input_ids and labels.
        
        Returns:
        --------
        dict
            A dictionary with the following keys:
            - input_ids: Tensor of padded input token IDs.
            - labels: Tensor of padded labels corresponding to the input data.
            - attention_mask: Tensor indicating which tokens should be attended to (non-padding tokens).
        """
        input_ids, labels = [], []
        for instance in instances:
            input_ids.append(torch.tensor(instance["input_ids"]))
            labels.append(torch.tensor(instance["labels"]))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )
