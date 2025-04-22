import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from main import reward_fn, format_reward, make_conversation, SYSTEM_PROMPT, get_training_dataset
from trl import GRPOConfig, GRPOTrainer

ds_train = load_dataset("demelin/moral_stories", "full", split='train')

# Process dataset to create prompts and ground truth labels
train_dataset = get_training_dataset(ds_train)

model_id = "Qwen/Qwen2.5-3B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="train_results",
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,
    # Parameters that control data preprocessing
    max_completion_length=1024,  # default: 256
    num_generations=4,  # default: 8
    max_prompt_length=256,  # default: 512
    # Parameters related to reporting and saving
    report_to=["wanndb"],
    logging_steps=10,
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=training_args,
    train_dataset=train_dataset,  # Use processed dataset
)

trainer.train()

trainer.save_model(training_args.output_dir)
trainer.push_to_hub(dataset_name="moral_norms")
