import re
from datasets import load_dataset
import random

SYSTEM_PROMPT = (
    """Reason about moral scenarios provided by the User. For each scenario:
1) Include your step-by-step reasoning in <think></think> tags
2) Identify the relevant norm in <norm></norm> tags
3) Provide your answer (A or B) in <answer></answer> tags"""
)

def check_answer(label, correct_label):
    label = label.strip().lower()
    correct_label = correct_label.strip().lower()
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


# check if answer is correct --> if so, return 1, else return 0
def reward_fn(completions, **kwargs):
    rewards = []
    completion_contents = [completion[0]["content"] for completion in completions]

    for completion, gt in zip(completion_contents, kwargs['ground_truth']):
        reward = 0
        reasoning = extract_text(completion, 'think')
        answer = extract_text(completion, 'answer')
        print(f"completion: {completion}")
        print(f"reasoning: {reasoning}, answer: {answer}, gt: {gt}")
        if reasoning is not None and answer is not None:
            reward = 0.1
            if check_answer(answer, gt):
                reward = 1
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
            {"role": "user", "content": f"""Scenario: {example["situation"]}.
Intention: {example["intention"]}.
Between the following two actions A and B, which one should be chosen?
A. {options[0]}
B. {options[1]}"""},
        ],
        "ground_truth": moral_label,
        # "id": example["id"]
    }



def get_training_dataset(ds_train):
    # filter out examples where either moral action or immoral action is "not specified"
    ds_train = ds_train.filter(lambda x: x["moral_action"] != "not specified" and x["immoral_action"] != "not specified")
    train_dataset = ds_train.map(make_conversation).select_columns(['prompt', 'ground_truth'])
    return train_dataset


if __name__ == "__main__":
    ds_train = load_dataset("demelin/moral_stories", "full", split='train')
    train_dataset = get_training_dataset(ds_train)
    print(train_dataset[0])
