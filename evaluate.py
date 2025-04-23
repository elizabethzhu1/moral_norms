import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import wandb
from utils import get_training_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import csv


def load_model_and_tokenizer(model_path, use_lora=False):
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def generate_completion(model, tokenizer, prompt_text, config):
    # Tokenize the prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    # Generate completion using config parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config['max_completion_length'],
            do_sample=True,
            temperature=config['temperature'],
            top_p=config['top_p'],
            min_p=config['min_p'],
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "assistant:" in completion.lower():
        completion = completion.lower().split("assistant:")[-1].strip()
    
    return completion


def extract_xml_tags(completion):
    try:
        # Extract answer
        answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
        answer = answer_match.group(1).strip().upper() if answer_match else None
        
        # Extract norm
        norm_match = re.search(r'<norm>(.*?)</norm>', completion, re.DOTALL)
        norm = norm_match.group(1).strip() if norm_match else None
        
        # Extract reasoning
        reasoning_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None
        
        return {
            'answer': answer,
            'norm': norm,
            'reasoning': reasoning,
            'valid': bool(answer and norm and reasoning)  # All required tags present
        }
    except Exception as e:
        print(f"Error extracting XML tags: {e}")
        return {
            'answer': None,
            'norm': None,
            'reasoning': None,
            'valid': False
        }


def calculate_similarity(ground_truth, generated):
    # Load a sentence transformer model for semantic similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode the sentences
    embeddings1 = model.encode([ground_truth])
    embeddings2 = model.encode([generated])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model on Hugging Face")
    parser.add_argument("--use_lora", action="store_true", help="Whether the model uses LoRA")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate on")
    args = parser.parse_args()

    config = json.load(open(args.config))
    
    # Initialize wandb for logging
    wandb.init(
        project="morals-evaluation",
        entity="cocolab",
        name=f"eval-{config['run_name']}-{args.split}",
        config=config,
    )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.use_lora)
    model.eval()

    # Load dataset
    ds_eval = load_dataset("demelin/moral_stories", "full", split=args.split)
    eval_dataset = get_training_dataset(ds_eval)

    results = {
        'correct_answers': 0,
        'norm_similarities': [],
        'total_samples': len(eval_dataset),
        'valid_completions': 0,
        'invalid_completions': 0
    }

    all_completions = []
    
    for i, example in enumerate(tqdm(eval_dataset[:len(eval_dataset)])):
        # Generate completion
        prompt_text = ""
        for message in example['prompt']:
            prompt_text += f"{message['role']}: {message['content']}\n"
            
        completion = generate_completion(model, tokenizer, prompt_text, config)
        all_completions.append(completion)
        # Extract XML tags
        extracted = extract_xml_tags(completion)
        
        if extracted['valid']:
            results['valid_completions'] += 1
            
            # Check if answer is correct
            if extracted['answer'] == example['ground_truth']:
                results['correct_answers'] += 1
                
            # Calculate norm similarity
            similarity = calculate_similarity(example['norm'], extracted['norm'])
            results['norm_similarities'].append(similarity)
        else:
            results['invalid_completions'] += 1
            results['norm_similarities'].append(0)  # 0 similarity for invalid completions
            
    # Calculate metrics
    results['accuracy'] = results['correct_answers'] / results['total_samples']
    results['avg_norm_similarity'] = np.mean(results['norm_similarities']) if results['norm_similarities'] else 0
    results['valid_completion_rate'] = results['valid_completions'] / results['total_samples']
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Average norm similarity: {results['avg_norm_similarity']:.4f}")
    print(f"Valid completion rate: {results['valid_completion_rate']:.4f}")

    # Save all completions to CSV (just in case)
    with open(f'completions_{args.split}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['completion'])  # header
        for completion in all_completions:
            writer.writerow([completion])
    
    # Log to wandb
    wandb.log({
        'accuracy': results['accuracy'],
        'avg_norm_similarity': results['avg_norm_similarity'],
        'valid_completion_rate': results['valid_completion_rate'],
    })
    
    # Save detailed results
    with open(f'evaluation_results_{args.split}.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

# TO RUN:
# For base model:
# python evaluate.py --model_path "your-username/your-model-name" --config config.json
