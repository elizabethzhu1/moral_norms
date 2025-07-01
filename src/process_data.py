#!/usr/bin/env python3
import os
import json
import csv
import sys
import utils

def load_all_results(root_dir, single=False):
    """
    Walks the immediate subdirectories of root_dir.
    For each subdirectory, if an 'all_results.json' file exists,
    it is loaded via json.load.
    Returns a dictionary mapping condition names (the subdirectory name)
    to a dictionary with keys:
      - "all_results": the parsed results from all_results.json
      - "dir": the full path to that subdirectory
    """
    results = {}
    if not single:
        for entry in os.listdir(root_dir):
            if entry.startswith("checkpoint-"):
                subdir = os.path.join(root_dir, entry)
                if os.path.isdir(subdir):
                    json_path = os.path.join(subdir, "all_results.json")
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f:
                                data = json.load(f)
                            results[entry] = {"all_results": data, "dir": subdir}
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON in {json_path}: {e}", file=sys.stderr)
                        except Exception as e:
                            print(f"Error reading {json_path}: {e}", file=sys.stderr)
                    else:
                        print(f"Warning: {json_path} does not exist.", file=sys.stderr)
    else:
        json_path = os.path.join(root_dir, "all_results.json")
        condition_str = root_dir.split('/')[-1]
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                results[condition_str] = {"all_results": data, "dir": root_dir}
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {json_path}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error reading {json_path}: {e}", file=sys.stderr)
        else:
            print(f"Warning: {json_path} does not exist.", file=sys.stderr)
    return results

def read_completions(completions_path):
    """
    Reads the completions file from completions_path.
    The expected format is a JSON array (or a JSON lines file that can be read with json.load).
    Returns the list of completion entries, or an empty list if reading fails.
    """
    if not os.path.exists(completions_path):
        print(f"Warning: Completions file {completions_path} does not exist.", file=sys.stderr)
        return []
    try:
        with open(completions_path, 'r') as f:
            # Try to load the file as JSON.
            completions = json.load(f)
        return completions
    except Exception as e:
        print(f"Error reading completions from {completions_path}: {e}", file=sys.stderr)
        return []

def compute_completions_metrics(completions):
    """
    Given a list of completion entries, computes:
      - average accuracy (from the "score" field)
      - average response length (number of words in the "generated" field)
    If a particular field is missing in an entry, that entry is skipped for that metric.
    Returns a tuple: (avg_accuracy, avg_response_length)
    """

    # TODO: compute accuracy and response length from model_responses.jsonl
    accurate = 0
    total = 0
    
    prompts = completions['prompt']
    responses = completions['completion']
    ground_truths = completions['ground_truth']
    dataset_names = completions['dataset_name']
    
    completion_lengths = []

    for prompt, response, ground_truth, dataset_name in zip(prompts, responses, ground_truths, dataset_names):

        answer = utils.extract_answer(response)
        if answer == ground_truth:
            accurate += 1
        total += 1
        completion_lengths.append(len(response))

    accuracy_rate = accurate / total
    avg_completion_length = sum(completion_lengths) / len(completion_lengths)

    print(f"Accuracy rate: {accuracy_rate}, Average completion length: {avg_completion_length}")
    
    return accuracy_rate, avg_completion_length



def process_data(results):
    """
    Given the collated results dictionary (mapping condition -> {"all_results": ..., "dir": ...}),
    iterate over each condition and each step (only include steps that are strings) in the condition's
    "steps_processed" list.
    
    For each such step:
      - Retrieve the metric values from the "results_by_step" dictionary.
      - Look for the corresponding completions file: model_responses.jsonl in that condition's directory.
      - Compute the average accuracy and average response length from the completions file.
      - Return a list of flattened row dictionaries.
    """
    rows = []
    for condition, info in results.items():
        cond_data = info.get("all_results", {})
        cond_dir = info.get("dir", "")
        steps = cond_data.get("steps_processed", [])
        results_by_step = cond_data.get("results_by_step", {})
        for step in steps:
            if not isinstance(step, str):
                continue  # Only process steps that are strings.
            if step not in results_by_step:
                print(f"Warning: step '{step}' not found in results_by_step for condition '{condition}'.", file=sys.stderr)
                continue
            # Get the metrics from the all_results.json
            metrics = results_by_step[step]
            # Build the base row.
            row = {"condition": condition, "step": step}
            row.update(metrics)
            # Now, locate and process the completions file.
            if int(step) == 1: 
                step = '0'
            completions_filename = f"model_responses.json"
            completions_path = os.path.join(cond_dir, completions_filename)
            completions = read_completions(completions_path)
            avg_acc, avg_resp_length = compute_completions_metrics(completions)
            row["accuracy"] = avg_acc
            row["response_length"] = avg_resp_length
            rows.append(row)

    return rows

def main():
    input_dir = '.'
    output_csv = os.path.join('.', 'flattened_results.csv')
    single = False

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # Collate the results from each condition.
    results = load_all_results(input_dir, single)
    if not results:
        print("No results were loaded. Check that your subdirectories contain all_results.json files.",
              file=sys.stderr)
        sys.exit(1)

    # Process the collated data (including completions metrics) to flatten it.
    rows = process_data(results)
    if not rows:
        print("No rows were produced. Check that your input data has steps_processed as strings.",
              file=sys.stderr)
        sys.exit(1)

    # Determine the fieldnames from the first row.
    fieldnames = list(rows[0].keys())
    # Write the flattened data to CSV.
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Wrote {len(rows)} rows to {output_csv}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

    # Write accuracy results for each checkpoint
    accuracy_results = {}
    for row in rows:
        condition = row['condition']
        step = row['step']
        accuracy = row['accuracy']
        response_length = row['response_length']
        if condition not in accuracy_results:
            accuracy_results[condition] = {}
        accuracy_results[condition][step] = {
            'accuracy': accuracy,
            'response_length': response_length
        }
    
    # Sort results by checkpoint number for each condition
    sorted_accuracy_results = {}
    # Sort the outer keys (conditions) numerically by checkpoint number
    def checkpoint_sort_key(name):
        try:
            return int(name.split('-')[1])
        except Exception:
            return float('inf')
    
    for condition in sorted(accuracy_results.keys(), key=checkpoint_sort_key):
        sorted_steps = sorted(accuracy_results[condition].keys(), key=lambda x: int(x))
        sorted_accuracy_results[condition] = {
            step: accuracy_results[condition][step] for step in sorted_steps
        }
   
    with open('accuracy_results.json', 'w') as f:
        json.dump(sorted_accuracy_results, f, indent=2)
    
    print(f"Wrote accuracy results to accuracy_results.json")

if __name__ == '__main__':
    main()
