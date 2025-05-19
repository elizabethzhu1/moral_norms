import argparse
import json
import sys
import pathlib

import torch
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator
import wandb

from src.utils import reward_fn, get_full_training_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.json")
args = parser.parse_args()

config = json.load(open(args.config))

accelerator = Accelerator()
if accelerator.is_main_process:
    wandb.init(
        project="morals",
        entity="cocolab",
        name=config["run_name"],
        config=config,
    )

train_dataset = get_full_training_dataset()

model_id = config['model_id']
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir=config['output_dir'],
    overwrite_output_dir=True,
    learning_rate=config['learning_rate'],
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    per_device_train_batch_size=config['per_device_train_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    num_train_epochs=config['num_train_epochs'],
    bf16=True,
    # Parameters that control data preprocessing
    max_completion_length=config['max_completion_length'],  # default: 256
    num_generations=config['num_generations'],  # default: 8
    max_prompt_length=config['max_prompt_length'],  # default: 512
    # Parameters related to reporting and saving
    report_to=["wandb"],
    logging_steps=config['logging_steps'],
    push_to_hub=False,
    save_strategy="steps",
    save_steps=config['save_steps'],
    gradient_checkpointing=True,
    torch_compile=True,
    temperature=config['temperature'],
    top_p=config['top_p'],
    min_p=config['min_p'],
    epsilon_high=0.28,
    epsilon=0.2,
    scale_rewards=False, # DR GRPO
    loss_type="dr_grpo",
    mask_truncated_completions=True,
    beta=config['beta'], # KL
    use_vllm=True,
    vllm_server_host=config['vllm_server_host'],
    vllm_server_port=8001,
    vllm_gpu_memory_utilization=0.95,
    vllm_enable_prefix_caching=True,
    )

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=train_dataset,  # Use processed dataset
)

trainer.train()
