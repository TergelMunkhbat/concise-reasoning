import os
import gc
import json
import torch
import argparse
from dataclasses import dataclass
from transformers.trainer_utils import set_seed
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs


from trainers.sft_trainer import SFTTrainingConfig, SFTTrainer
from dataset import GSM8kDatasetLoader, MATHDatasetLoader
from training_utils import format_zero_shot_prompt, generate_responses
from prompt import LLAMA_CHAT_TEMPLATE
from model import load_model, load_model_with_flash_attention, get_latest_checkpoint
from utils import convert_to_json, filter_rationales


@dataclass
class IterativeTrainingConfig(SFTTrainingConfig):
    """Configuration for Expert Iteration training"""
    
    # Iteration specific configs
    iteration_method: str = 'shift'
    num_iterations: int = 3
    start_step: int = 0
    
    # Generation configs
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 40 
    top_p: float = 0.95
    num_diverse_paths: int = 1
    prompt_system: str = "irpo"
    accelerate: bool = True
    
    # Filtering configs
    only_correct: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        # Create iteration data directory
        self.iteration_data_dir = os.path.join(self.output_dir, self.iteration_method)
        os.makedirs(self.iteration_data_dir, exist_ok=True)


class IterativeTrainer(SFTTrainer):
    """Expert Iteration Trainer"""
    
    def __init__(self, **kwargs):
        self.config = IterativeTrainingConfig(**kwargs)
        self.current_iteration = 0
        self.current_iter_dir = None
        self.dataset_loader = None
        self.original_dataset = None
        self._create_iteration_directories()
        
        # Initialize accelerator for distributed generation
        if self.config.accelerate:
            kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
            self.accelerator = Accelerator(kwargs_handlers=[kwargs])
        else:
            self.accelerator = None
    
    def _create_iteration_directories(self):
        """Ensure all necessary directories exist"""
        os.makedirs(self.config.iteration_data_dir, exist_ok=True)
        
        for i in range(self.config.num_iterations):
            iter_dir = os.path.join(self.config.iteration_data_dir, f"iteration_{i}")
            os.makedirs(iter_dir, exist_ok=True)
            os.makedirs(os.path.join(iter_dir, "raw"), exist_ok=True)
            os.makedirs(os.path.join(iter_dir, "filtered"), exist_ok=True)
            os.makedirs(os.path.join(iter_dir, "ckpts"), exist_ok=True)
            os.makedirs(os.path.join(iter_dir, "logs"), exist_ok=True)
            os.makedirs(os.path.join(iter_dir, "results"), exist_ok=True)
    
    def _update_current_iter_dir(self):
        self.current_iter_dir = os.path.join(
            self.config.iteration_data_dir,
            f"iteration_{self.current_iteration}"
        )
        self.config.output_dir = self.current_iter_dir
    
    def _format_zero_shot(self, example):
        return format_zero_shot_prompt(example, self.config.prompt_system, self.config.model_name)
    
    def _apply_chat_template(self, example):
        if "llama" in self.config.model_name:
            return {
                "prompt": self.tokenizer.apply_chat_template(
                    example['messages'],
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=LLAMA_CHAT_TEMPLATE
                )
            }
        else:
            return {
                "prompt": self.tokenizer.apply_chat_template(
                    example['messages'],
                    tokenize=False,
                    add_generation_prompt=True
                )
            }
    
    def clear_cache(self):
        """Clear model from memory"""
        if self.logger:
            self.logger.info("Starting to clear cache...")
            if hasattr(self, 'model'):
                self.logger.info("Clearing model from memory")
            if hasattr(self, 'tokenizer'):
                self.logger.info("Clearing tokenizer from memory")
        
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        # Log memory stats after clearing
        if self.logger and torch.cuda.is_available():
            self.logger.info(f"GPU memory allocated after clearing: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
            self.logger.info(f"GPU memory cached after clearing: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")
            self.logger.info("Cache clearing completed")
    
    def load_model(self, iteration, type="train"):
        """Load model"""
        if iteration == 0:
            model_path = self.config.model_path
        else:
            iter_dir = os.path.join(
                self.config.iteration_data_dir,
                f"iteration_{iteration-1}",
                "ckpts"
            )
            
            model_path = get_latest_checkpoint(iter_dir)
            
        if self.logger:
            self.logger.info(f"Loading model from {model_path}")

        # Load model and tokenizer
        if self.config.use_flash_attention:
            if self.config.accelerate and type == "test":
                self.model, self.tokenizer = load_model_with_flash_attention(
                model_path,
                {"": self.accelerator.process_index},
                type
            )
            else:
                self.model, self.tokenizer = load_model_with_flash_attention(
                    self.config.model_path,
                    "auto",
                    type
                )
        else:
            if self.config.accelerate and type == "test":
                self.model, self.tokenizer = load_model(
                    model_path, 
                    {"": self.accelerator.process_index},
                    type
                )
            else:
                self.model, self.tokenizer = load_model(
                    self.config.model_path, 
                    "auto", 
                    type
                )
    
    def setup(self) -> None:
        """Initialize model, tokenizer, datasets"""
        # Set random seed
        set_seed(self.config.seed)
        
        # Update current iteration directory
        self._update_current_iter_dir()
        
        if self.logger:
            self.logger.info(f"Setting up iteration {self.current_iteration}...")
        
        # Initialize dataset loader if not yet
        if self.dataset_loader is None:
            if self.config.dataset == 'gsm8k':
                self.dataset_loader = GSM8kDatasetLoader()
            elif self.config.dataset == 'math':
                self.dataset_loader = MATHDatasetLoader()
            else:
                raise ValueError(f"Unsupported dataset: '{self.config.dataset}'")
            self.dataset_loader.set_logger(self.logger)
        
        # Load original dataset if not done yet
        if self.original_dataset is None:
            if self.logger:
                self.logger.info("Loading original datasets...")
            datasets = self.dataset_loader.load_from_json()
            self.original_dataset = datasets['train']
              
        # Generate new training data for this iteration
        if self.logger:
            self.logger.info(f"Generating data for iteration {self.current_iteration}")
        
        generated_data_dir = self.generate_data()
        
        self.train_dataset = self.dataset_loader.load_llm_preds(
            generated_data_dir,
            self.config.type,
            split='train',
            percentage=self.config.percentage,
            target_total=self.config.target_total
        )
        
        # Clean up training datasets
        self.train_dataset = self.train_dataset.remove_columns(['length', 'gold'])
        
        # Tokenize training datasets
        self.train_dataset = self.train_dataset.map(
            self.tokenize_function,
            remove_columns=['input', 'label']
        )
        
        if self.logger:
            self.logger.info(f"Setup completed")
            self.logger.info(f"Train dataset size: {len(self.train_dataset)}")
    
    def generate_data(self):
        """Generate new training data using current model"""
        args = argparse.Namespace(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            batch_size = self.config.batch_size*8,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )
        
        # Format dataset for generation
        if self.current_iteration == 0:
            if self.logger:
                self.logger.info("Formatting dataset with zero-shot prompts")
            # For first iteration, use instruction format
            generation_dataset = self.original_dataset.map(self._format_zero_shot)
            # Apply chat template
            generation_dataset = generation_dataset.map(self._apply_chat_template)
            generation_dataset = generation_dataset.remove_columns(["messages"])
        else:
            if self.logger:
                self.logger.info("Using direct prompts")
            # For later iterations, use direct format
            generation_dataset = self.original_dataset.map(self._format_direct_prompt)
        
        # Duplicate each example num_diverse_paths times
        if self.config.num_diverse_paths > 1:
            if self.logger:
                self.logger.info(f"Duplicating dataset {self.config.num_diverse_paths} times for diverse paths")
            indices = [i for i in range(len(generation_dataset)) for _ in range(self.config.num_diverse_paths)]
            generation_dataset = generation_dataset.select(indices)
            
        if self.logger:
            self.logger.info("Generating responses...")
            
        # Generate responses
        outputs, output_token_counts = generate_responses(
            generation_dataset,
            self.model, 
            self.tokenizer, 
            args
        )
        
        # Save raw generations to jsonl
        raw_output_dir = os.path.join(self.current_iter_dir, f"raw")
        
        raw_file_path = convert_to_json(
            generation_dataset,
            outputs,
            output_token_counts,
            raw_output_dir
        )
        
        if self.logger:
            self.logger.info(f"Saved raw generations to {raw_file_path}")
            self.logger.info("Reading and filtering generations...")
        
        # Read back and filter
        with open(raw_file_path, 'r') as f:
            raw_data = [json.loads(line) for line in f if line.strip()]
        
        if self.logger:
            self.logger.info("Filtering generations...")
            
        # Filter rationales
        filtered_data = filter_rationales(
            raw_data,
            max_tokens=self.config.max_new_tokens,
            only_correct=self.config.only_correct,
            logger=self.logger
        )
        
        filtered_file_dir = self.save_filtered_rationales(
            filtered_data,
            dir_path=os.path.join(self.current_iter_dir, "filtered")
        )
        
        return filtered_file_dir
        
    def save_filtered_rationales(self, data, dir_path='./'):
        "Save rationales to JSON file"
        os.makedirs(f"{dir_path}/train", exist_ok=True)
        
        # Format data for saving
        data_to_save = [
            {
                'input': item['input'],
                'label': item['label'],
                'rationale': item['rationale'],
                'token_count': item['token_count'],
                'dataset': item['dataset'],
                '_id': item['_id']
            }
            for item in data
        ]
        
        # Save to a file
        filename = f"{dir_path}/train/filtered_correct_rationales.json"
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        if self.logger:
            self.logger.info(f"Saved {len(data_to_save)} rationales to '{filename}'")
        
        return dir_path
        
    
    def train(self):
        """Run iterative training loop"""
        for iteration in range(self.config.num_iterations):
            self.current_iteration = iteration
            
            if self.logger and torch.cuda.is_available():
                self.logger.info(f"Starting iteration {iteration}")
                self.logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
                    
            if iteration > 0:
                self.clear_cache()
            
            # Load model for  generation
            self.load_model(iteration, "test")    
            
            if self.logger and torch.cuda.is_available():
                self.logger.info(f"GPU memory after model loading: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
                
            # Setup for this iteration
            self.setup()
            
            # Clear and reload model for training
            self.clear_cache()
            self.load_model(iteration, "train")
            
            if self.logger and torch.cuda.is_available():
                self.logger.info(f"GPU memory after model loading: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
            
            # Train using parent's training logic
            super().train()
            
            if self.logger:
                self.logger.info(f"Completed iteration {iteration}")
            