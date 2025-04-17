import re
from datasets import load_dataset
import random

def check_answer(answer, correct_answer):
    # FIX THIS TO DO AN ACTUAL COMPARISON WITH LLM
    return True


def extract_text(text, tag):
    pattern = fr'<{tag}>(.*?)</{tag}>'
    return re.findall(pattern, text, re.DOTALL)

# parse <think>, <norm>, <answer>
def reward_fn(result, correct_answer, correct_norm):
    result = [0, 0]  # [norm, answer]
    reasoning = extract_text('think')
    norm = extract_text('norm')
    answer = extract_text('answer')

    print("REASONING:", reasoning)
    print("NORM:", norm)
    print("ANSWER:", answer)

    print("CORRECT NORM:", correct_norm)
    print("CORRECT ANSWER:", correct_answer)

    if check_answer(norm, correct_norm):
        result[0] = 1
    else:
        result[0] = 0

    # check if answer is correct via LLM comparison
    if check_answer(answer, correct_answer):
        result[1] = 1
    else:
        result[1] = 0

    return result


prompts = []
def main():
    ds = load_dataset("demelin/moral_stories", "full")

    print(ds['train'][0])

    # format as multiple choice questions between moral + immoral actions
    for i in range(len(ds['train'])):
        story = ds['train'][i]['story']
        moral_action = ds['train'][i]['moral_action']
        immoral_action = ds['train'][i]['immoral_action']

        moral_label = random.choice(['A', 'B'])
        immoral_label = 'B' if moral_label == 'A' else 'A'

        prompt = f"""
            Scenario: {story}
            Between the following two actions A and B, which one should be chosen?
            {moral_label}. Moral Action: {moral_action}
            {immoral_label}. Immoral Action: {immoral_action}
            """
        
        prompts.append(prompt)
    
    print("PROMPTS:", prompts)


if __name__ == '__main__':
    main()
