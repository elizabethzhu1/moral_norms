import json
import re
import numpy as np
from scipy import stats

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_thinking_length(completion):
    # Extract text between <think> and </think> tags
    match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    if match:
        return len(match.group(1).strip())
    return 0

def analyze_relationships(data):
    # Initialize lists to store data
    thinking_lengths = []
    correct_answers = []
    norm_similarities = []
    correct_norms = []
    
    # Extract data from each response
    for response in data:
        thinking_length = extract_thinking_length(response['completion'])
        thinking_lengths.append(thinking_length)
        correct_answers.append(response['correct'])
        norm_similarities.append(response['norm_similarity']['rating'])
        correct_norms.append(response['valid_norm'])
    
    # Convert to numpy arrays
    thinking_lengths = np.array(thinking_lengths)
    correct_answers = np.array(correct_answers)
    norm_similarities = np.array(norm_similarities)
    correct_norms = np.array(correct_norms)
    
    # Calculate correlation between thinking length and correct answers
    corr_length_correct, p_value_length = stats.pearsonr(thinking_lengths, correct_answers)
    
    # Calculate P(B|A) where B is correct answer and A is high norm similarity (>=5)
    high_similarity = norm_similarities >= 5
    p_high_similarity = np.mean(high_similarity)
    p_correct_and_high_similarity = np.mean(correct_answers & high_similarity)
    p_correct_given_high_similarity = p_correct_and_high_similarity / p_high_similarity if p_high_similarity > 0 else 0
    
    # Print results
    print("Analysis Results:")
    print(f"Correlation between thinking length and correct answers: {corr_length_correct:.3f} (p-value: {p_value_length:.3f})")
    print(f"Probability of correct answer given high norm similarity (>=5): {p_correct_given_high_similarity:.3f}")
    
    # Additional analysis: correlation between norm similarity and correct answers
    corr_similarity_correct, p_value_similarity = stats.pearsonr(norm_similarities, correct_answers)
    print(f"Correlation between norm similarity and correct answers: {corr_similarity_correct:.3f} (p-value: {p_value_similarity:.3f})")

if __name__ == "__main__":
    data = load_data("valid_responses.json")
    analyze_relationships(data) 