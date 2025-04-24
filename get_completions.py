import requests
import json
import argparse
from typing import Optional, Dict, Any, Tuple
from datasets import load_dataset
from utils import get_training_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from utils import make_conversation, SYSTEM_PROMPT


def extract_user_prompt(conversation: Dict[str, Any]) -> str:
    """
    Extract the user prompt from the conversation format.
    The conversation format from utils.py is:
    {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "..."}
        ],
        "ground_truth": "..."
    }
    """
    for message in conversation["prompt"]:
        if message["role"] == "user":
            return message["content"]
    return ""


def generate_completion(system_prompt: str, user_prompt: str, config: Dict[str, Any], llm: LLM) -> Optional[str]:
    """
    Generate a completion using vLLM with proper conversation formatting.
    """
    # Format the complete prompt to match training format
    complete_prompt = f"""SYSTEM: {system_prompt}
                    USER: {user_prompt}
                    ASSISTANT:"""
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=config['max_completion_length'],
        temperature=config['temperature'],
        top_p=config['top_p'],
        min_p=config['min_p'],
        stop=["USER:"]  # Stop generation at the next user turn
    )
    
    try:
        # Generate completion
        outputs = llm.generate(complete_prompt, sampling_params)
        completion = outputs[0].outputs[0].text.strip()
        return completion
    except Exception as e:
        print(f"Error generating completion: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate completions using vLLM')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (Hugging Face repo or local path)')
    parser.add_argument('--output', type=str, default='model_responses.json', help='Output file path')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
        
    # Initialize vLLM
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=config.get('vllm_gpu_memory_utilization', 0.95),
        enable_prefix_caching=config.get('vllm_enable_prefix_caching', True),
    )
        
    # Load evaluation dataset
    ds_eval = load_dataset("demelin/moral_stories", "full", split='test')
    eval_dataset = get_training_dataset(ds_eval)
    
    # Generate completions
    results = []
    for i, example in enumerate(tqdm(eval_dataset)):
        # Get the conversation format from utils.py
        conversation = make_conversation(example)
        user_prompt = extract_user_prompt(conversation)
        ground_truth = conversation["ground_truth"]
            
        # Generate completion
        completion = generate_completion(SYSTEM_PROMPT, user_prompt, config, llm)
        
        # Store result
        result = {
            'example_id': i,
            'system_prompt': SYSTEM_PROMPT,
            'user_prompt': user_prompt,
            'completion': completion,
            'ground_truth': ground_truth,
            'norm': example['norm']
        }
        results.append(result)
        
        if completion:
            print(f"\nExample {i+1}/{len(eval_dataset)}:")
            print(f"User Prompt: {user_prompt}")
            print(f"Completion: {completion}")
        else:
            print(f"\nFailed to generate completion for example {i+1}/{len(eval_dataset)}")
            
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} completions to {args.output}")

if __name__ == "__main__":
    main()
