import json
import argparse
from typing import Optional, Dict, Any
from datasets import load_dataset, concatenate_datasets
from utils import get_eval_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import make_conversation
import torch

def get_eval_dataset():
    ds_eval_moral_stories = load_dataset("demelin/moral_stories", "gen-norm$actions+context+consequences-norm_distance", split='test')
    ds_eval_scruples = load_dataset("metaeval/scruples", split='test')
    ds_eval_ethics_commensense = load_dataset("hendrycks/ethics", "commonsense", split='test')
    ds_eval_ethics_deontology = load_dataset("hendrycks/ethics", "deontology", split='test')
    ds_eval_ethics_justice = load_dataset("hendrycks/ethics", "justice", split='test')
    ds_eval_utilitarianism = load_dataset("hendrycks/ethics", "utilitarianism", split='test')

    # concatenate all datasets
    ds_eval = concatenate_datasets([ds_eval_moral_stories, ds_eval_scruples, ds_eval_ethics_commensense, ds_eval_ethics_deontology, ds_eval_ethics_justice, ds_eval_utilitarianism])

    return ds_eval
    

def generate_completion(conversation: str, config: Dict[str, Any], model, tokenizer) -> Optional[str]:
    try:
        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and move to device
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Use torch.inference_mode() to disable autograd overhead
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config['max_completion_length'],
                temperature=config['temperature'],
                top_p=config['top_p'],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache
                num_beams=1,  # Use greedy decoding for speed
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract text after 'assistant'
        assistant_start = generated_text.find('assistant\n')
        if assistant_start != -1:
            completion = generated_text[assistant_start + len('assistant'):].strip()
        else:
            completion = generated_text  # Fallback if 'assistant' not found

        return completion
    except Exception as e:
        print(f"Error generating completion: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate completions using local model')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--output', type=str, default='model_responses.json', help='Output file path')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
        
    # Check if MPS is available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
        
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("./clean_morals_mvp")
    model = AutoModelForCausalLM.from_pretrained(
        "./clean_morals_mvp",
        device_map=device,  # Use MPS if available
        torch_dtype=torch.bfloat16,  
        low_cpu_mem_usage=True,     # Optimize memory usage
        trust_remote_code=True
    )

    # Set model to evaluation mode and enable optimizations
    model.eval()
    
    # Load evaluation dataset
    eval_dataset = get_eval_dataset()
    
    # Generate completions
    results = []
    for i, example in enumerate(tqdm(eval_dataset)):
        # Get the conversation format from utils.py
        conversation = make_conversation(example)
        prompt = conversation["prompt"]
        ground_truth = conversation["ground_truth"]
            
        # Generate completion
        completion = generate_completion(prompt, config, model, tokenizer)
        
        # Store result
        result = {
            'example_id': i,
            'prompt': prompt,
            'completion': completion,
            'ground_truth': ground_truth,
            'norm': example['norm'],
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
