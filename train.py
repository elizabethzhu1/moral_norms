from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from main import reward_fn

dataset = load_dataset("demelin/moral_stories", "full")

training_args = GRPOConfig(output_dir="train_results", logging_steps=10)
trainer = GRPOTrainer(
    model="gpt-4o-mini",
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=dataset['train'],
)
trainer.train()
