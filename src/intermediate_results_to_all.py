#!/usr/bin/env python3
import json
import os
import glob
from typing import Dict, Any


def get_step_files(parent_dir: str) -> list[str]:
    """
    Recursively get all evaluation step files from all checkpoint-* subdirectories in parent_dir.
    """
    step_files = []
    for entry in os.listdir(parent_dir):
        entry_path = os.path.join(parent_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith("checkpoint-"):
            pattern = os.path.join(entry_path, "evaluation_step*.json")
            step_files.extend(glob.glob(pattern))
    
    step_files = [(step_file.split("step")[-1].split(".")[0], step_file) for step_file in step_files]
    step_files = sorted(step_files, key=lambda x: int(x[0]))

    return step_files
    

def aggregate_evaluation_results(output_dir: str) -> None:
    """
    Aggregate all evaluation step results into a single all_results.json file.
    Works with GPT evaluation results from gpt_api_eval.py.
    
    Args:
        output_dir: Directory containing the evaluation step files
    """
    # Initialize the aggregate results structure
    aggregate_results = {
        "results_by_step": {},
        "steps_processed": []
    }
    
    # Initialize norms collection for separate file
    all_norms = {}
    
    # Find all evaluation step files
    step_files = get_step_files(output_dir)
    
    # Process each step file
    for step, step_file in step_files:
        try:
            # Read the step results
            with open(step_file, 'r') as f:
                step_data = json.load(f)
            
            # Extract the results for this step
            step_results = step_data.get('results', {})
            
            # Initialize metrics for this step
            step_metrics = {
                'values': {'care': [], 'fairness': [], 'loyalty': [], 'authority': [], 'purity': []},
                'moral_framework': {'utilitarianism': [], 'deontology': [], 'justice': [], 'virtue_ethics': [], 'care_ethics': []},
                'counterfactual': [],
                'norm': [],
                'development': [],
                'motivation': [],
                'total_problems': len(step_results)
            }
            
            # Collect norms for this step
            step_norms = []
            
            # Process each problem in the step
            for problem_id, problem_data in step_results.items():
                # Extract values
                if 'values' in problem_data:
                    values_metrics = problem_data['values']['metrics']
                    for value_type in ['care', 'fairness', 'loyalty', 'authority', 'purity']:
                        if value_type in values_metrics and values_metrics[value_type] is not None:
                            step_metrics['values'][value_type].append(int(values_metrics[value_type]))

                
                # Extract moral framework
                if 'moral_framework' in problem_data:
                    framework_metrics = problem_data['moral_framework']['metrics']
                    for framework_type in ['utilitarianism', 'deontology', 'justice', 'virtue_ethics', 'care_ethics']:
                        if framework_type in framework_metrics and framework_metrics[framework_type] is not None:
                            step_metrics['moral_framework'][framework_type].append(int(framework_metrics[framework_type]))
                
                # Extract counterfactual
                if 'counterfactual' in problem_data:
                    counterfactual_val = problem_data['counterfactual']['metrics'].get('counterfactual')
                    if counterfactual_val is not None:
                        step_metrics['counterfactual'].append(int(counterfactual_val))
                
                # Extract norm
                if 'norm' in problem_data:
                    norm_val = problem_data['norm']['metrics'].get('norm')
                    if norm_val is not None:
                        step_metrics['norm'].append(norm_val)
                        step_norms.append(norm_val)
                
                # Extract development
                if 'development' in problem_data:
                    development_val = problem_data['development']['metrics'].get('development')
                    if development_val is not None:
                        step_metrics['development'].append(development_val)
                
                # Extract motivation
                if 'motivation' in problem_data:
                    motivation_val = problem_data['motivation']['metrics'].get('motivation')
                    if motivation_val is not None:
                        step_metrics['motivation'].append(int(motivation_val))
            
            # Store norms for this step
            all_norms[step] = step_norms
            
            # Calculate averages for numeric metrics
            step_summary = {
                'total_problems': step_metrics['total_problems'],
                'values_distribution': {},
                'moral_framework_distribution': {},
                'counterfactual_distribution': {},
                'motivation_distribution': {},
                'development_distribution': {}
            }
            
            # Calculate value distributions
            for value_type, values in step_metrics['values'].items():
                step_summary['values_distribution'][value_type] = {}
                for score in range(1, 8):  # 1 to 7
                    step_summary['values_distribution'][value_type][str(score)] = values.count(score)
            
            # Calculate moral framework distributions
            for framework_type, values in step_metrics['moral_framework'].items():
                step_summary['moral_framework_distribution'][framework_type] = {}
                for score in range(1, 8):  # 1 to 7
                    step_summary['moral_framework_distribution'][framework_type][str(score)] = values.count(score)
            
            # Calculate counterfactual distribution
            for score in range(1, 8):  # 1 to 7
                step_summary['counterfactual_distribution'][str(score)] = step_metrics['counterfactual'].count(score)
            
            # Calculate motivation distribution
            for score in range(1, 8):  # 1 to 7
                step_summary['motivation_distribution'][str(score)] = step_metrics['motivation'].count(score)
            
            # Calculate development distribution
            development_counts = {}
            for dev in step_metrics['development']:
                development_counts[dev] = development_counts.get(dev, 0) + 1
            step_summary['development_distribution'] = development_counts
            
            # Add to aggregate results
            aggregate_results['results_by_step'][step] = step_summary
            
        except Exception as e:
            print(f"Error processing {step_file}: {str(e)}")
            continue
    
    # Update steps processed list
    numeric_steps = [s for s in aggregate_results['results_by_step'].keys() 
                    if s.isdigit()]
    aggregate_results['steps_processed'] = sorted(numeric_steps, key=int)
    
    # Save the aggregate results
    output_path = os.path.join('.', "all_results.json")
    with open(output_path, 'w') as f:
        json.dump(aggregate_results, f, indent=2)
    
    print(f"Aggregated results saved to: {output_path}")
    print(f"Processed {len(aggregate_results['steps_processed'])} steps")

    # Save norms to separate file
    norms_path = os.path.join(".", "extracted_norms.json")
    with open(norms_path, 'w') as f:
        json.dump(all_norms, f, indent=2)
    
    print(f"Norms saved to: {norms_path}")
if __name__ == "__main__":
    aggregate_evaluation_results('.')

