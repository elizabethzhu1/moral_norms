import json
import os
import argparse
import pandas as pd
from utils import get_full_eval_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description='Generate completions using VLLM')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--output', type=str, default='model_responses.csv', help='Output file path')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling parameter')
    parser.add_argument('--save_dir', type=str, default='model_responses', help='Directory to save the completions')
    args = parser.parse_args()

    # Initialize VLLM model
    model = LLM(
        model=args.model_path,
        tokenizer_mode="auto",
        max_num_seqs=32,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        max_num_batched_tokens=2048,
        enable_chunked_prefill=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Load evaluation dataset
    eval_dataset = get_full_eval_dataset()
    
    # Prepare data for generation
    prompts = eval_dataset['prompt']
    ground_truths = eval_dataset['ground_truth']
    dataset_names = eval_dataset['dataset_name']

    outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    # Generate completions in batches
    result = {
        'prompt': prompts,
        'completion': responses,
        'ground_truth': ground_truths,
        'dataset_name': dataset_names
    }

    # save
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'model_responses.json'), 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
