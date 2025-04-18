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


# check if answer is correct --> if so, return 1, else return 0
def reward_fn(result, correct_answer):
    reward = 0

    reasoning = extract_text(result, 'think')
    answer = extract_text(result, 'answer')

    print("REASONING:", reasoning)
    print("ANSWER:", answer)


    if check_answer(answer, correct_answer):
        reward += 1
    
    return reward


"""Reward function that checks if the completion has a specific format.
This function appears redundant since we already have a more comprehensive reward_fn above 
that not only checks format but also validates answers and provides partial rewards."""
def format_reward(completions, **kwargs):
    # This function is likely not needed since:
    # 1. reward_fn already checks format via extract_text()
    # 2. reward_fn provides more granular rewards (0.5 for each component)
    # 3. reward_fn validates answer correctness
    # 4. reward_fn is the one actually used in train.py
    pattern = r"^<think>.*?</think>\s*<norm>.*?</norm>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]


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
        "ground_truth": moral_label
    }


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. As the assistant, you will be given hypothetical scenarios in which a subject finds themselves in. You are to reason step-by-step about what the morally appropriate action is for the subject given this scenario. Enclose your step by step reasoning process between a starting <think> tag and a </think> tag marking the end, i.e., <think> reasoning process here </think>. Then, identify a relevant norm that informs the appropriate action for the subject to take and specify this norm between the tags <norm> and </norm>. Finally, specify the correct answer by enclosing it between the tags <answer> and </answer>. The answer should be one of the options A or B."
)

# generate MC questions from dataset
prompts = []
def main():
    ds_train = load_dataset("demelin/moral_stories", "full", split='train')

    train_dataset = ds_train.map(make_conversation).select_columns(['prompt', 'ID'])

    print(train_dataset[0])


if __name__ == '__main__':
    main()
