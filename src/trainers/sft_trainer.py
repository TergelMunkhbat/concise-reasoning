import os
from datetime import datetime
from dataclasses import dataclass
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import set_seed


from model import load_model, load_model_with_flash_attention, DataCollatorForSupervisedDataset
from dataset import GSM8kDatasetLoader, MATHDatasetLoader, MMLUProDatasetLoader


IGNORE_INDEX = -100

@dataclass
class SFTTrainingConfig:
    # Dataset configuration
    dataset: str
    model_path: str
    model_name: str
    output_dir: str
    type: str
    data_path: str = './data'
    percentage: float = 100
    target_total: int = None
    
    # Training hyperparameters
    batch_size: int = 16
    grad_steps: int = 1
    learning_rate: float = 1e-5
    seed: int = 0
    
    # Training process configurations
    num_train_epochs: float = 1.0
    save_strategy: str = "epoch"
    save_steps: float = 1.0
    logging_steps: int = 16
    save_total_limit: int = 3
    lr_scheduler_type: str = "constant"
    
    # Hardware and logging configurations
    bf16: bool = False
    no_log: bool = False
    
    # Generation configs
    max_new_tokens: int = 512
    
    # Flash Attention specific configurations
    use_flash_attention: bool = False

    # Whether to use the raw output directory or the organized one
    use_raw_output_dir: bool = False
    
    def __post_init__(self):
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        if not self.use_raw_output_dir:
            self.output_dir = os.path.join(
                self.output_dir,
                self.dataset,
                self.model_name,
                self.type,
                f"{self.percentage}",
                timestamp
        )
        
        # Ensure bf16 is enabled when using Flash Attention
        if self.use_flash_attention and not self.bf16:
            self.bf16 = True

class SFTTrainer:
    def __init__(self, **kwargs):
        self.config = SFTTrainingConfig(**kwargs)
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.logger = None
    
    def set_logger(self, logger):
        """Set logger for the trainer"""
        self.logger = logger
    
    def _format_direct_prompt(self, example):
        if "qwen" in self.config.model_name.lower():
            return {"prompt": example['input']}
        else:
            return {"prompt": self.tokenizer.bos_token + example['input']}
    
    def setup(self) -> None:
        """Initialize model, tokenizer, and datasets"""
        # Set random seed
        set_seed(self.config.seed)
        
        if self.logger:
            self.logger.info("Setting up trainer...")
            self.logger.info(f"Loading model from {self.config.model_path}")
        
        # Load model and tokenizer
        if self.config.use_flash_attention:
            self.model, self.tokenizer = load_model_with_flash_attention(
                self.config.model_path,
                "auto",
                "train"
            )
        else:
            self.model, self.tokenizer = load_model(
                self.config.model_path, 
                "auto", 
                "train"
            )
        
        # Load datasets
        if self.config.dataset == 'gsm8k':
            dataset_loader = GSM8kDatasetLoader()
        elif self.config.dataset == 'math':
            dataset_loader = MATHDatasetLoader()
        elif self.config.dataset.startswith('mmlu-pro-'):
            category = self.config.dataset[len('mmlu-pro-'):]
            dataset_loader = MMLUProDatasetLoader(category=category)
        else:
            raise ValueError(f"Unsupported dataset: '{self.config.dataset}'")
    
        dataset_loader.set_logger(self.logger)
        
        if self.logger:
            self.logger.info("Loading training datasets...")
        
        # Load training dataset
        self.train_dataset = dataset_loader.load_llm_preds(
            self.config.data_path, 
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
            self.logger.info("Setup completed")
    
    def tokenize_function(self, example):
        # Creat input part with BOS
        if "qwen" in self.config.model_name.lower():
            input_text = example['input']
        else:
            input_text = self.tokenizer.bos_token + example['input']
        
        # Tokenize the input portion to get its length
        input_tokenized = self.tokenizer(input_text, add_special_tokens=False)
        input_length = len(input_tokenized['input_ids'])
        
        # Create full text by adding label and EOS token
        full_text = input_text + ' ' + example['label'] + self.tokenizer.eos_token

        # Tokenize complete text
        model_inputs = self.tokenizer(full_text, add_special_tokens=False)
        
        # Create labels with IGNORE_INDEX for input portion
        labels = model_inputs["input_ids"].copy()
        labels[:input_length] = [IGNORE_INDEX] * input_length

        model_inputs['labels'] = labels
        model_inputs['prompt'] = input_text

        return model_inputs
    
        
    def train(self) -> None:
        """Run the training process"""
        if self.model is None:
            self.setup()
        
        if self.logger:
            self.logger.info("Starting training process...")
            
        # Set output directories
        output_dir = f"{self.config.output_dir}/ckpts"
        logging_dir = f"{self.config.output_dir}/logs" if not self.config.no_log else None
        
        if self.logger:
            self.logger.info(f"Output directory: {output_dir}")
            self.logger.info(f"Logging directory: {logging_dir}")
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir,
            remove_unused_columns=False,
            logging_dir=logging_dir,
            
            # Training duration
            num_train_epochs=self.config.num_train_epochs,
            
            # Learning rate schedule configuration
            lr_scheduler_type = self.config.lr_scheduler_type,
            
            # Batch size and optimization
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.grad_steps,
            learning_rate=self.config.learning_rate,
            
            # saving strategy
            save_strategy=self.config.save_strategy,
            logging_strategy="steps" if not self.config.no_log else "no",
            save_steps=self.config.save_steps if self.config.save_strategy == "steps" else None,
            logging_steps=self.config.logging_steps,
            
            # Model saving
            save_total_limit=self.config.save_total_limit,
            
            # Other settings
            seed=self.config.seed,
            bf16=self.config.bf16,
            report_to='tensorboard',
        )
        
        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        
        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Start training
        trainer.train()
        
        if self.logger:
            self.logger.info("Training completed")
            