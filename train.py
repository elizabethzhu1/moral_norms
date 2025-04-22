import argparse
import json

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator
import wandb
from main import reward_fn, format_reward, make_conversation, SYSTEM_PROMPT, get_training_dataset

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

ds_train = load_dataset("demelin/moral_stories", "full", split='train')

# Process dataset to create prompts and ground truth labels
train_dataset = get_training_dataset(ds_train)

model_id = config['model_id']
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# lora_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
#     target_modules=["q_proj", "v_proj"],
# )

# model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

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
    use_vllm=True,
    gradient_checkpointing=True,
    torch_compile=True,
    temperature=config['temperature'],
    top_p=config['top_p'],
    min_p=config['min_p'],
    epsilon_high=0.28,
    epsilon_low=0.2,
    scale_rewards=False, # DR GRPO
    beta=config['beta'], # KL
    vllm_server_host=config['vllm_server_host'],
    vllm_server_port=8000,
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

trainer.save_model(training_args.output_dir)
trainer.push_to_hub(dataset_name="moral_norms")
