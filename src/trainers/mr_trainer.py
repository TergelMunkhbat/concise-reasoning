import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from dataset import Dataset
import torch.nn.functional as F
from accelerate.utils import broadcast_object_list


from utils import convert_to_json, filter_rationales
from training_utils import generate_responses
from trainers.iterative_trainer import IterativeTrainer
from math_parser import compare_answers


class MRTrainer(IterativeTrainer):
    """Metareasoning Trainer"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_rewards(self, batch_data):
        """Calculate rewards according to paper's formulation"""
        self.logger.info("Computing rewards")
        
        # Group data by _id to handle multiple rationales for same input
        df = pd.DataFrame(batch_data)
        grouped = df.groupby('_id')
        
        all_rewards = []
        processed_data = {
            '_id': [], 'input': [], 'label': [], 
            'rationale': [], 'token_count': [], 
            'dataset': [], 'reward': []
        }
        
        # Process each group (problem) separately
        for _id, group in tqdm(grouped, desc="Computing rewards", total=len(grouped)):
            # Format data for computation
            input_text = group['input'].iloc[0]
            label = group['label'].iloc[0]
            rationales = group['rationale'].tolist()
            
            # Format inputs for model
            prompts = [self.format_input(input_text) for _ in rationales]
            formatted_rationales = [self.format_rationale(r) for r in rationales]
            formatted_labels = [self.format_label(label) for _ in rationales]
            
            # Add empty rationale case for (π(y|x))
            prompts.append(self.format_input(input_text))
            formatted_rationales.append("")
            formatted_labels.append(self.format_label(label))
            
            # Tokenize
            inputs = self._batch_tokenize(prompts, formatted_rationales, formatted_labels)
            
            # Compute probabilities
            with torch.no_grad():
                outputs = self.model.forward(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    return_dict=True
                )
                logits = outputs.logits[:, :-1, :].contiguous()
            
            # Get probabilities for each sequence
            target = inputs['input_ids'][:, 1:]
            target_mask = inputs['target_mask'][:, 1:]
            logprobs = self.logprobs_from_logits(logits, target)
            probabilities = torch.tensor([torch.exp(lp[m.bool()]).sum() for lp, m in zip(logprobs, target_mask)]).to(self.device)
            
            # Split probabilities into rationales and baseline
            baseline_prob = probabilities[-1]  # π(y|x) from empty rationale
            rationale_probs = probabilities[:-1]  # π(y|z,x) from rationales
            
            # Calculate utility: U(z|x,y) = log π(y|z,x) - log π(y|x)
            epsilon = 1e-4  # For numerical stability
            utilities = torch.log((rationale_probs + epsilon) / (baseline_prob + epsilon))
            
            # Calculate cost: C(z) = γ * log l(z)
            rationale_lengths = inputs['thoughts_mask'][:-1].sum(dim=1).float()  # Exclude baseline
            costs = 0.1 * torch.log(rationale_lengths + 1.0)
            
            # Calculate final rewards: R(x,y,z) = U(z|x,y) - C(z)
            rewards = utilities - costs
            
            # Store results
            for idx, (reward, rationale) in enumerate(zip(rewards, rationales)):
                processed_data['_id'].append(_id)
                processed_data['input'].append(input_text)
                processed_data['label'].append(label)
                processed_data['rationale'].append(rationale)
                processed_data['token_count'].append(group['token_count'].iloc[idx])
                processed_data['dataset'].append(group['dataset'].iloc[idx])
                processed_data['reward'].append(reward.item())
                all_rewards.append(reward.item())
            
            # Clear memory
            del outputs, logits, probabilities, utilities, costs, rewards
            torch.cuda.empty_cache()
        
        # Log reward statistics
        rewards_tensor = torch.tensor(all_rewards)
        self.logger.info(f"Rewards statistics:")
        self.logger.info(f"Mean reward: {rewards_tensor.mean().item():.4f}")
        self.logger.info(f"Max reward: {rewards_tensor.max().item():.4f}")
        self.logger.info(f"Min reward: {rewards_tensor.min().item():.4f}")
        
        return processed_data
    
    def _batch_tokenize(self, prompts, rationales, labels):
        """Tokenize and format a batch of sequences"""
        # Combine sequences
        full_sequences = [f"{p}{r}{l}" for p, r, l in zip(prompts, rationales, labels)]
        
        # Get lengths for masking
        prompt_lengths = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        rationale_lengths = [len(self.tokenizer.encode(r, add_special_tokens=False)) for r in rationales]
        label_lengths = [len(self.tokenizer.encode(l, add_special_tokens=False)) for l in labels]
        
        # Tokenize full sequences
        tokenized = self.tokenizer(
            full_sequences,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        
        # Create masks
        seq_length = tokenized['input_ids'].size(1)
        batch_size = len(prompts)
        thoughts_mask = torch.zeros((batch_size, seq_length), dtype=torch.long)
        target_mask = torch.zeros((batch_size, seq_length), dtype=torch.long)
        
        # Fill masks
        for i in range(batch_size):
            start_rationale = prompt_lengths[i]
            end_rationale = start_rationale + rationale_lengths[i]
            start_label = end_rationale
            end_label = start_label + label_lengths[i]
            
            thoughts_mask[i, start_rationale:end_rationale] = 1
            target_mask[i, start_label:end_label] = 1
        
        return {
            'input_ids': tokenized['input_ids'].to(self.device),
            'attention_mask': tokenized['attention_mask'].to(self.device),
            'thoughts_mask': thoughts_mask.to(self.device),
            'target_mask': target_mask.to(self.device)
        }
            
    def format_input(self, text: str) -> str:
        """Format input text"""
        return self.tokenizer.bos_token + 'Question: ' + text if self.tokenizer.bos_token else text
    
    def format_rationale(self, rationale: str) -> str:
        """Format rationale text"""
        return f"\nSolution:{rationale}\n"
    
    def format_label(self, label: str) -> str:
        """Format label text"""
        return f"\nTherefore, the final answer is {label}" + self.tokenizer.eos_token
    
    def logprobs_from_logits(self, logits: torch.Tensor, labels: torch.Tensor, gather: bool = True):
        logp = F.log_softmax(logits, dim=2)
        if not gather:
            return logp
        logpy = logp.gather(2, labels.unsqueeze(2)).squeeze(-1)
        return logpy
    
    def save_filtered_rationales(self, data, dir_path='./'):
        """Save rationales with rewards to JSON file"""
        # Calculate rewards before saving
        data_with_rewards = self.compute_rewards(data)
        
        os.makedirs(f"{dir_path}/train", exist_ok=True)
        
        # Convert parallel lists into list of dictionaries with native Python types
        data_to_save = []
        for i in range(len(data_with_rewards['_id'])):
            item_dict = {
                'input': str(data_with_rewards['input'][i]),
                'label': str(data_with_rewards['label'][i]),
                'rationale': str(data_with_rewards['rationale'][i]),
                'token_count': int(data_with_rewards['token_count'][i]),
                'dataset': str(data_with_rewards['dataset'][i]),
                '_id': str(data_with_rewards['_id'][i]),
                'reward': float(data_with_rewards['reward'][i])
            }
            data_to_save.append(item_dict)
        
        filename = f"{dir_path}/train/filtered_rationales.json"
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        if self.logger:
            self.logger.info(f"Saved {len(data_to_save)} rationales with rewards to '{filename}'")
        
        return dir_path
        
    def _create_dataset_with_hints(self, raw_data):
        """Add hints to questions with incorrect answers"""
        if self.logger:
            self.logger.info("Adding hints to incorrect answers for second generation attempt")
        
        # Create dataset for incorrect answers
        incorrect_data = []
        incorrect_indices = []
        
        for i, example in enumerate(raw_data):
            # Check if answer was incorrect
            # Check if answer was incorrect based on dataset type
            is_correct = (
                compare_answers(example['input'], example['label'], example['answer'])
                if example['dataset'] == 'math'
                else example['answer'] == example['label']
            )
            
            if not is_correct:
                # Create new example with modified input
                hint_str = f"\n(Hint: The correct answer is {example['label']})"
                
                incorrect_data.append({
                    'input': example['input'] + hint_str,
                    'label': example['label'],
                    'dataset': example['dataset'],
                    '_id': example['_id']  # Preserve ID for tracking
                })
                incorrect_indices.append(i)
                
        if self.logger:
            self.logger.info(f"Found {len(incorrect_data)} incorrect answers for regeneration")
        
         # Format the dataset
        if self.current_iteration == 0:
             # For first iteration, use instruction format
            generation_dataset = Dataset.from_list(incorrect_data).map(self._format_zero_shot)
            generation_dataset = generation_dataset.map(self._apply_chat_template)
            generation_dataset = generation_dataset.remove_columns(["messages"])
        else:
            generation_dataset = Dataset.from_list(incorrect_data).map(self._format_direct_prompt)
        
        return generation_dataset, incorrect_indices
    
    def _merge_generations(self, original_raw_data, retry_raw_data, incorrect_indices):
        """Merge original and retry generations, keeping better answers"""
        merged_data = original_raw_data.copy()
        
        for new_item, orig_idx in zip(retry_raw_data, incorrect_indices):
            # Check if retry was correct
            is_correct = (
                compare_answers(new_item['input'], new_item['label'], new_item['answer'])
                if new_item['dataset'] == 'math'
                else new_item['answer'] == new_item['label']
            )
            if is_correct:
                merged_data[orig_idx] = new_item
                
        return merged_data
    
    def _check_existing_raw_data(self):
        """Check if raw data already exists and read all files"""
        raw_output_dir = os.path.join(self.current_iter_dir, "raw")
        raw_data = []
        
        # Get all jsonl files
        files = [f for f in os.listdir(raw_output_dir) if f.endswith('.json')]
        
        if not files:
            if self.logger:
                self.logger.info("No existing raw data found")
            return None
        
        try:
            # Read all files
            for filename in files:
                file_path = os.path.join(raw_output_dir, filename)
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            raw_data.append(json.loads(line))
            
            if raw_data:
                if self.logger:
                    self.logger.info(f"Loaded {len(raw_data)} examples from {len(files)} files")
                return raw_data
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error loading raw data: {str(e)}")
            return None
                    
    def generate_data(self):
        """Generate new training data using current model with distributed processing"""
        
        # Check for existing filtered data first
        filtered_file_dir = None
        if not self.accelerator or self.accelerator.is_main_process:
            filtered_file_dir = self._check_filtered_data_exists()
        
        if self.accelerator:
            filtered_file_dir = broadcast_object_list([filtered_file_dir])[0]
            self.accelerator.wait_for_everyone()
        
        if filtered_file_dir is not None:
            if self.logger:
                self.logger.info("Using existing filtered data instead of generating new data")
            return filtered_file_dir
        
        # Check for existing raw data first
        raw_data = self._check_existing_raw_data()
        
        if raw_data is not None:
            if self.logger:
                self.logger.info("Using existing raw data instead of generating new data")
        else:
            args = argparse.Namespace(
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                batch_size = self.config.batch_size*4,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )
            
            # Calculate progressive sample size
            total_samples = len(self.original_dataset)
            current_samples = int(total_samples * ((self.current_iteration + 1) / self.config.num_iterations))
            
            if self.logger:
                self.logger.info(f"Using {current_samples} samples for iteration {self.current_iteration}")
            
            # Format dataset for generation
            if self.current_iteration == 0:
                if self.logger:
                    self.logger.info("Formatting dataset with zero-shot prompts")
                # For first iteration, use instruction format
                generation_dataset = self.original_dataset.map(self._format_zero_shot)
                generation_dataset = generation_dataset.map(self._apply_chat_template)
                generation_dataset = generation_dataset.remove_columns(["messages"])
            else:
                if self.logger:
                    self.logger.info("Using direct prompts")
                # For later iterations, use direct format
                generation_dataset = self.original_dataset.map(self._format_direct_prompt)
                
            # Limit dataset size based on current iteration
            generation_dataset = generation_dataset.select(range(current_samples))
            
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
                args,
                accelerator=self.accelerator
            )
            
            raw_file_path = None
            raw_output_dir = os.path.join(self.current_iter_dir, f"raw")
            # Save raw generations to jsonl (only main process if using accelerator)
            if not self.accelerator or self.accelerator.is_main_process:
                
                raw_file_path = convert_to_json(
                    generation_dataset,
                    outputs,
                    output_token_counts,
                    raw_output_dir
                )
                
                if self.logger:
                    self.logger.info(f"Saved raw generations to {raw_file_path}")
                    self.logger.info("Reading and filtering generations...")
            
            if self.accelerator:
                raw_file_path = broadcast_object_list([raw_file_path])[0]
                self.accelerator.wait_for_everyone()
            
            # Read back and filter
            with open(raw_file_path, 'r') as f:
                raw_data = [json.loads(line) for line in f if line.strip()]
            
            # Second generation attempt for incorrect answers
            retry_dataset, incorrect_indices = self._create_dataset_with_hints(raw_data)
            
            if len(retry_dataset) > 0:
                if self.logger and (not self.accelerator or self.accelerator.is_main_process):
                    self.logger.info("Generating second round responses with hints...")
                
                # Generate retry responses
                retry_outputs, retry_token_counts = generate_responses(
                    retry_dataset,
                    self.model,
                    self.tokenizer,
                    args,
                    accelerator=self.accelerator
                )
                
                # Save retry generations
                if not self.accelerator or self.accelerator.is_main_process:
                    retry_file_path = convert_to_json(
                        retry_dataset,
                        retry_outputs,
                        retry_token_counts,
                        raw_output_dir
                    )

                    # Read retry raw data
                    with open(retry_file_path, 'r') as f:
                        retry_raw_data = [json.loads(line) for line in f if line.strip()]
                
                    # Merge results, keeping the better version of each answer
                    raw_data = self._merge_generations(raw_data, retry_raw_data, incorrect_indices)
        
        if self.logger and (not self.accelerator or self.accelerator.is_main_process):
            self.logger.info("Filtering generations...")
            
        # Filter rationales (only main process if using accelerator)
        filtered_file_dir = None
        if not self.accelerator or self.accelerator.is_main_process:
            filtered_data = filter_rationales(
                raw_data,
                max_tokens=self.config.max_new_tokens,
                only_correct=self.config.only_correct,
                dataset_type=self.config.dataset,
                logger=self.logger
            )
        
            filtered_file_dir = self.save_filtered_rationales(
                filtered_data,
                dir_path=os.path.join(self.current_iter_dir, "filtered")
            )
        
        if self.accelerator:
            filtered_file_dir = broadcast_object_list([filtered_file_dir])[0]
            self.accelerator.wait_for_everyone()
        
        return filtered_file_dir

    def check_iteration_completed(self, iteration):
        """Check if an iteration has already been completed"""
        iter_dir = os.path.join(self.config.iteration_data_dir, f"iteration_{iteration}")
        checkpoint_dir = os.path.join(iter_dir, "ckpts")
        
        # Check if checkpoints directory exists and contains checkpoints
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint-')]
            if checkpoints:  # If any checkpoints exist
                if self.logger:
                    self.logger.info(f"Iteration {iteration} already has checkpoint(s), skipping...")
                    self.logger.info(f"Found checkpoints: {checkpoints}")
                return True
        return False
    
    def _check_filtered_data_exists(self):
        """Check if filtered data already exists"""
        filtered_dir = os.path.join(self.current_iter_dir, "filtered")
        filtered_file = os.path.join(filtered_dir, "train", "filtered_rationales.json")
        
        if os.path.exists(filtered_file):
            if self.logger:
                self.logger.info(f"Found existing filtered data at {filtered_file}")
            return filtered_dir
        return None
        
    def train(self):
        for iteration in range(self.config.num_iterations):
            self.current_iteration = iteration
            
            if self.check_iteration_completed(iteration):
                continue
            
            if self.logger and torch.cuda.is_available():
                self.logger.info(f"Starting iteration {iteration}")
                self.logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
                    
            if iteration > 0:
                self.clear_cache()
                
            # Load modal for generation
            if self.accelerator:
                # Generation phase
                if self.logger:
                    self.logger.info("Running in generation mode")
                
                self.load_model(iteration, "test")
                self.setup()  # This will generate data
                if self.logger:
                    self.logger.info("Generation completed. Exiting.")
                return
            else:
                # Training phase
                if self.logger:
                    self.logger.info("Running in training mode")
                
                self.load_model(iteration=0, type="train")
                self.setup()
                # Train using parent's training logic
                super(IterativeTrainer, self).train()
                
                if self.logger:
                    self.logger.info(f"Completed iteration {iteration}. Exiting.")
                return
            