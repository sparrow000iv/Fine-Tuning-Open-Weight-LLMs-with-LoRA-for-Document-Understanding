"""
Training script for LoRA fine-tuning of Mistral-7B on document understanding tasks.

Honestly this took way longer to get right than I expected. The main issues were:
1. Memory management - kept OOMing until I switched to 8-bit AdamW
2. NaN losses with fp16 - switched to bf16 and it was fine
3. The data loading was a pain - HuggingFace datasets expects specific formats

Usage:
    python scripts/train.py --config configs/mistral_lora_r16.yaml
"""

import os
import json
import yaml
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Suppress some annoying HF warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.WARNING)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_and_prepare_data(data_path, tokenizer, max_length=2048):
    """
    Load training data and format it for SFT.
    
    Expected JSON format (one per line):
    {"instruction": "...", "input": "...", "output": "..."}
    """
    examples = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                # Format as a chat-style prompt (Mistral Instruct format)
                formatted = f"<s>[INST] {ex['instruction']}\n\n{ex['input']} [/INST] {ex['output']}</s>"
                examples.append({"text": formatted})
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed line: {line[:100]}...")
                continue
    
    logger.info(f"Loaded {len(examples)} training examples")
    
    dataset = Dataset.from_list(examples)
    
    # Quick sanity check - print a few examples
    for i in range(min(3, len(dataset))):
        logger.info(f"Example {i}: {dataset[i]['text'][:200]}...")
    
    return dataset


def load_and_prepare_eval_data(data_path, tokenizer, max_length=2048):
    """Load eval data in the same format as training."""
    if not os.path.exists(data_path):
        logger.warning(f"No eval data found at {data_path}, skipping eval")
        return None
    return load_and_prepare_data(data_path, tokenizer, max_length)


def get_model_and_tokenizer(config):
    """
    Load the base model and tokenizer.
    
    I went back and forth on whether to use the Instruct or base variant.
    Instruct seems to work better for structured output tasks like this.
    """
    model_name = config['model']['name']
    use_qlora = config['model'].get('use_qlora', False)
    
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Make sure pad token is set (Mistral doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = "right"  # Important for causal LMs
    
    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,  # bf16 worked way better than fp16 for me
    }
    
    if use_qlora:
        logger.info("Using QLoRA (4-bit quantization)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        # For regular LoRA, load in bf16
        # If you're OOMing here, switch to use_qlora=True in config
        model_kwargs["device_map"] = "auto"
    
    logger.info(f"Loading model from {model_name} (this might take a while...)")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


def get_lora_config(config):
    """Build LoRA config from YAML config."""
    lora_cfg = config['lora']
    
    # Target modules - I found targeting all linear layers was worth the slight slowdown
    target_modules = lora_cfg.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    
    lora_config = LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['alpha'],
        lora_dropout=lora_cfg.get('dropout', 0.05),
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    
    return lora_config


def print_trainable_params(model):
    """Print the number of trainable parameters."""
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return trainable, total


def train(config_path):
    """Main training function."""
    
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    # Create output directory
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the config used for this run (for reproducibility)
    with open(os.path.join(output_dir, 'train_config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Set random seed for reproducibility
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(config)
    
    # Apply LoRA
    lora_config = get_lora_config(config)
    model = get_peft_model(model, lora_config)
    print_trainable_params(model)
    
    # Load data
    train_data = load_and_prepare_data(
        config['data']['train_path'],
        tokenizer,
        config['training'].get('max_seq_length', 2048)
    )
    
    eval_data = load_and_prepare_eval_data(
        config['data'].get('eval_path', ''),
        tokenizer,
        config['training'].get('max_seq_length', 2048)
    )
    
    # Training arguments
    train_cfg = config['training']
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get('epochs', 3),
        per_device_train_batch_size=train_cfg.get('batch_size', 4),
        gradient_accumulation_steps=train_cfg.get('gradient_accumulation', 8),
        learning_rate=float(train_cfg.get('learning_rate', 2e-4)),
        weight_decay=train_cfg.get('weight_decay', 0.01),
        warmup_steps=train_cfg.get('warmup_steps', 100),
        lr_scheduler_type=train_cfg.get('scheduler', 'cosine'),
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,  # Keep only 3 best checkpoints
        eval_strategy="steps" if eval_data else "no",
        eval_steps=200 if eval_data else None,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # This saved me from OOM so many times
        max_seq_length=train_cfg.get('max_seq_length', 2048),
        report_to="wandb" if config.get('use_wandb', False) else "none",
        run_name=config.get('experiment_name', 'lora_doc_understanding'),
        seed=seed,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )
    
    # Train!
    logger.info("Starting training...")
    logger.info(f"Total training steps: {len(train_data) // (train_cfg.get('batch_size', 4) * train_cfg.get('gradient_accumulation', 8)) * train_cfg.get('epochs', 3)}")
    
    train_result = trainer.train()
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model')
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Save training metrics
    metrics = train_result.metrics
    with open(os.path.join(output_dir, 'train_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Training complete! Final loss: {metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"Model saved to: {final_model_path}")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LoRA-adapted LLM for document understanding')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    
    train(args.config)
