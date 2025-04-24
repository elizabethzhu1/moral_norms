import requests
import json
import argparse
from typing import Optional, Dict, Any
from datasets import load_dataset
from utils import get_training_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from utils import make_conversation, SYSTEM_PROMPT
from transformers import AutoTokenizer


def generate_completion(conversation: list, config: Dict[str, Any], llm: LLM, tokenizer: AutoTokenizer) -> Optional[str]:
    """
    Generate a completion using vLLM with proper chat template formatting.
    """
    # Apply chat template to format the conversation
    formatted_prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=config['max_completion_length'],
        temperature=config['temperature'],
        top_p=config['top_p'],
        min_p=config['min_p'],
        stop=tokenizer.eos_token  # Stop at end of sequence
    )
    
    try:
        # Generate completion
        outputs = llm.generate(formatted_prompt, sampling_params)
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
        
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
    # Initialize vLLM with chat template configuration
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=config.get('vllm_gpu_memory_utilization', 0.95),
        enable_prefix_caching=config.get('vllm_enable_prefix_caching', True),
        chat_template=tokenizer.chat_template,  # Pass the chat template to vLLM
        trust_remote_code=True,  # Required for some chat templates
    )
        
    # Load evaluation dataset
    ds_eval = load_dataset("demelin/moral_stories", "full", split='test')
    eval_dataset = get_training_dataset(ds_eval)
    
    # Generate completions
    results = []
    for i, example in enumerate(tqdm(eval_dataset)):
        # Get the conversation format from utils.py
        conversation = make_conversation(example)
        prompt = conversation["prompt"]
        ground_truth = conversation["ground_truth"]
            
        # Generate completion
        completion = generate_completion(prompt, config, llm, tokenizer)
        
        # Store result
        result = {
            'example_id': i,
            'prompt': prompt,
            'completion': completion,
            'ground_truth': ground_truth,
            'norm': example['norm']
        }
        results.append(result)
        
        if completion:
            print(f"\nExample {i+1}/{len(eval_dataset)}:")
            print(f"Prompt: {prompt}")
            print(f"Completion: {completion}")
        else:
            print(f"\nFailed to generate completion for example {i+1}/{len(eval_dataset)}")
            
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} completions to {args.output}")

if __name__ == "__main__":
    main()
