import re
from datasets import load_dataset
import random

SYSTEM_PROMPT = (
    """Reason about the following moral scenario provided by the User. For each scenario, you must provide ALL of the following in order:

        1) Your step-by-step reasoning between <think> and </think> tags
        2) The relevant moral norm (a standard of right or wrong behavior that is widely accepted within our society) between <norm> and </norm> tags
        3) Your final answer (ONLY 'A' or 'B') between <answer> and </answer> tags

        Your response MUST follow the following format:
        <think>
        [Your detailed reasoning here]
        </think>
        <norm>
        [The relevant moral norm]
        </norm>
        <answer>
        A
        </answer>
    """
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
        if "</answer>" in completion:
            completion = completion.split("</answer>")[0] + "</answer>"
        else:
            rewards.append(0)
            continue
        reasoning = extract_text(completion, 'think')
        answer = extract_text(completion, 'answer')
        print(f"completion: {completion}")
        print()
        print(f"reasoning: {reasoning}, answer: {answer}, gt: {gt}")
        if reasoning is not None and answer is not None:
            reward = 0.1
            if check_answer(answer, gt):
                reward = 1
        rewards.append(reward)
    return rewards


def make_conversation_moral_stories(example):
    # Remove unnecessary columns
    example = {k: v for k, v in example.items() if k in ['situation', 'intention', 'moral_action', 'immoral_action']}

    moral_label = random.choice(['A', 'B'])
    
    if moral_label == 'A':
        options = [example["moral_action"], example["immoral_action"]]
    else:
        options = [example["immoral_action"], example["moral_action"]]

    # should we modify to be formatted with xml tags like how we're evaluating?
    return {
        f"prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Scenario: {example["situation"]}.
            Intention: {example["intention"]}.
            Between the following two actions A and B, which one should be chosen?
            A. {options[0]}
            B. {options[1]}"""},
        ],
        "ground_truth": moral_label
    }


def make_conversation_scruples(example):
    # Remove unnecessary columns
    example = {k: v for k, v in example.items() if k in ['action', 'text', 'post_type', 'binared_label']}

    ground_truth_label = random.choice(['A', 'B'])

    # In dataset, 0 is 'Right' and 1 is 'Wrong'
    ground_truth = example['binared_label']
    
    if ground_truth == 'Right':
        not_ground_truth = 'Wrong'
    else:
        not_ground_truth = 'Right'
    
    if ground_truth_label == 'A':
        options = [ground_truth, not_ground_truth]
    else:
        options = [not_ground_truth, ground_truth]

    # should we modify to be formatted with xml tags like how we're evaluating?
    return {
        f"prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Scenario: {example['text']}.
            Action: {example['action']}
            Between the following two judgments A and B, which one reflects the author's action?
            A. {options[0]}
            B. {options[1]}"""},
        ],
        "ground_truth": ground_truth
    }


def make_conversation_ethics(example):
    # Remove unnecessary columns
    example = {k: v for k, v in example.items() if k in ['label', 'input']}

    ground_truth_label = random.choice(['A', 'B'])

    # In dataset, 0 is 'Right' and 1 is 'Wrong'
    if example['label'] == 0:
        ground_truth = 'Right'
        not_ground_truth = 'Wrong'
    else:
        ground_truth = 'Wrong'
        not_ground_truth = 'Right'
    
    if ground_truth_label == 'A':
        options = [ground_truth, not_ground_truth]
    else:
        options = [not_ground_truth, ground_truth]
    
    return {
        f"prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Scenario: {example['input']}.
            Between the following two judgments A and B, which one reflects this action?
            A. {options[0]}
            B. {options[1]}"""},
        ],
        "ground_truth": ground_truth_label
    }


def get_training_dataset(ds_train):
    # filter out examples where either moral action or immoral action is "not specified"
    ds_train = ds_train.filter(lambda x: x["moral_action"] != "not specified" and x["immoral_action"] != "not specified")

    train_dataset_ethics = ds_train.map(make_conversation_ethics).select_columns(['prompt', 'ground_truth'])
    train_dataset_moral = ds_train.map(make_conversation_moral_stories).select_columns(['prompt', 'ground_truth'])
    train_dataset_scruples = ds_train.map(make_conversation_scruples).select_columns(['prompt', 'ground_truth'])

    train_dataset = train_dataset_ethics + train_dataset_moral + train_dataset_scruples

    return train_dataset


def get_eval_dataset(ds_eval):
    eval_dataset = ds_eval.filter(lambda x: x["moral_action"] != "not specified" and x["immoral_action"] != "not specified")
    return eval_dataset


if __name__ == "__main__":
    ds_train = load_dataset("demelin/moral_stories", "full", split='train')
    train_dataset = get_training_dataset(ds_train)
    print(train_dataset[0])
