import json
import argparse
import pandas as pd
from utils import get_full_eval_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description='Generate completions using VLLM')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--output', type=str, default='model_responses.csv', help='Output file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling parameter')
    args = parser.parse_args()

    # Initialize VLLM model
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=1,  # Adjust based on your GPU setup
        gpu_memory_utilization=0.9,
        trust_remote_code=True
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

    # Generate completions in batches
    results = []
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i:i + args.batch_size]
        
        # Generate completions for the batch
        outputs = model.generate(batch_prompts, sampling_params)
        
        # Process results
        for j, output in enumerate(outputs):
            idx = i + j
            result = {
                'prompt': batch_prompts[j],
                'completion': output.outputs[0].text,
                'ground_truth': ground_truths[idx],
                'dataset_name': dataset_names[idx]
            }
            results.append(result)
            
            if j < 2:  # Print first few examples for monitoring
                print(f"\nExample {idx+1}/{len(prompts)}:")
                print(f"Prompt: {batch_prompts[j]}")
                print(f"Completion: {output.outputs[0].text}")
    
    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(results)} completions to {args.output}")


if __name__ == "__main__":
    main()
