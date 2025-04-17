import re
from datasets import load_dataset
import random

def check_answer(answer, correct_answer):
    # FIX THIS TO DO AN ACTUAL COMPARISON WITH LLM
    return True


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
    reasoning = extract_text(result, 'think')
    answer = extract_text(result, 'answer')

    print("REASONING:", reasoning)
    print("ANSWER:", answer)

    if check_answer(answer, correct_answer):
        return 1
    else:
        return 0


def make_conversation(example):
    # Remove unnecessary columns
    example = {k: v for k, v in example.items() if k in ['situation', 'moral_action', 'immoral_action']}

    moral_label = random.choice(['A', 'B'])
    immoral_label = 'B' if moral_label == 'A' else 'A'
    
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
        "moral_label": moral_label
    }


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


# generate MC questions from dataset
prompts = []
def main():
    ds_train = load_dataset("demelin/moral_stories", "full", split='train')

    train_dataset = ds_train.map(make_conversation).select_columns(['prompt', 'ID'])

    print(train_dataset[0])


if __name__ == '__main__':
    main()
