import re
from datasets import load_dataset
import random

def check_answer(label, correct_label):
    if label == correct_label:
        return True
    return False


def extract_text(text, tag):
    if f'<{tag}>' not in text:
        return None

    pattern = fr'<{tag}>(.*?)</{tag}>'
    matches = re.findall(pattern, text, re.DOTALL)

    # if xml tag is not found, return None
    if not matches:
        return None

    try:
        final_match = matches[-1].strip()
        return final_match
    except:
        return None

def reward_fn(completions, **kwargs):
    rewards = []
    for completion in completions:

        reasoning = extract_text(completion['text'], 'think')
        answer = extract_text(completion['text'], 'answer')

        if check_answer(answer, kwargs.example['ground_truth']):
            reward = 1
        else:
            if reasoning is not None and answer is not None:
                reward += 0.1
            else:
                reward = 0

        rewards.append(reward)
    
    return rewards


def make_conversation(example):
    # Remove unnecessary columns
    example = {k: v for k, v in example.items() if k in ['situation', 'moral_action', 'immoral_action']}

    moral_label = random.choice(['A', 'B'])
    
    if moral_label == 'A':
        options = [example["moral_action"], example["immoral_action"]]
    else:
        options = [example["immoral_action"], example["moral_action"]]

    return {
        f"prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Scenario: {example["situation"]}
            Between the following two actions A and B, which one should be chosen?
            A. {options[0]}
            B. {options[1]}"""},
        ],
        "ground_truth": moral_label,
        # "id": example["id"]
    }


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. As the assistant, you will be given hypothetical scenarios in which a subject finds themselves in. You are to reason step-by-step about what the morally appropriate action is for the subject given this scenario. Enclose your step by step reasoning process between a starting <think> tag and a </think> tag marking the end, i.e., <think> reasoning process here </think>. Then, identify a relevant norm that informs the appropriate action for the subject to take and specify this norm between the tags <norm> and </norm>. Finally, specify the correct answer by enclosing it between the tags <answer> and </answer>. The answer should be one of the options A or B."
)

def get_training_dataset(ds_train):
    train_dataset = ds_train.map(make_conversation).select_columns(['prompt', 'ground_truth'])
    return train_dataset


if __name__ == "__main__":
    ds_train = load_dataset("demelin/moral_stories", "full", split='train')
    train_dataset = get_training_dataset(ds_train)
    print(train_dataset[0])
