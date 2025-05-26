import argparse
import json
from tqdm import tqdm
# import wandb
import numpy as np
import re
from typing import Optional, Dict, Any
from openai import OpenAI


def extract_xml_tags(completion):
    # Extract answer - allow for spaces and capitalization
    answer_match = re.search(r'<\s*[Aa][Nn][Ss][Ww][Ee][Rr]\s*>(.*?)</\s*[Aa][Nn][Ss][Ww][Ee][Rr]\s*>', completion, re.DOTALL)
    answer = answer_match.group(1).strip().upper() if answer_match else None
    
    # Extract reasoning - allow for spaces and capitalization
    reasoning_match = re.search(r'<\s*[Tt][Hh][Ii][Nn][Kk]\s*>(.*?)</\s*[Tt][Hh][Ii][Nn][Kk]\s*>', completion, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    
    return {
        'answer': answer,
        'reasoning': reasoning,
        'valid_answer': bool(answer),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--responses", type=str, default="model_responses.json", help="Path to model responses file")
    parser.add_argument("--openai_key", type=str, required=True, help="OpenAI API key")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Initialize OpenAI client
    client = OpenAI(api_key=args.openai_key)

    # Load model responses from model_responses.json (or arg provided)
    with open(args.responses) as f:
        responses = json.load(f)

    results = {
        'correct_answers': 0,
        'total_samples': len(responses),
        'valid_answers': 0,
    }
    
    # Initialize list to store all prompts and completions
    all_responses = []
    
    for response in tqdm(responses):
        completion = response['completion']
        if completion is None:
            continue
            
        # Extract XML tags
        extracted = extract_xml_tags(completion)
        
        # Store prompt and completion
        response_data = {
            'prompt': response['prompt'],
            'completion': completion,
            'correct_answer': response['ground_truth'],
            'extracted_answer': extracted['answer'],
            'valid_answer': extracted['valid_answer'],
        }
        
        if extracted['valid_answer']:
            results['valid_answers'] += 1
            response_data['valid_answer'] = True

            print(f"Extracted answer: {extracted['answer']}")
            print(f"Ground truth: {response['ground_truth']}")

            # Check if answer is correct
            if extracted['answer'] == response['ground_truth']:
                results['correct_answers'] += 1
                response_data['correct'] = True
            else:
                response_data['correct'] = False

        else:
            response_data['valid_answer'] = False
            response_data['correct'] = False
            response_data['norm_similarity'] = None

        all_responses.append(response_data)
        
    # Save all prompts and completions
    with open('all_responses.json', 'w') as f:
        json.dump(all_responses, f, indent=2)
            
    # Calculate metrics
    results['accuracy'] = results['correct_answers'] / results['valid_answers'] if results['valid_answers'] > 0 else 0
    
    results['valid_answer_rate'] = results['valid_answers'] / results['total_samples']
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Valid answer rate: {results['valid_answer_rate']:.4f}")
    
    # Save detailed results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
