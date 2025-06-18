import json
import tempfile
from openai import OpenAI
import asyncio
import argparse
import os
import random
import re
import fcntl
from utils import extract_values, extract_norm, extract_counterfactual, extract_moral_framework, extract_development, extract_motivation

class BaseAnalyzer:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)
    
    def create_analysis_prompts(self, numbers, target, completion):
        """Create the analysis prompts for a given problem."""
        pass

    def convert_to_batch_format(self, input_path, num_samples=200):
        """
        Convert arithmetic problems to OpenAI Batch API format with all analysis prompts.
        If input_path is a directory, combine all JSON files whose names match
        the pattern 'completion_step<d+>' (e.g. completion_step3.json). Each item is
        augmented with a 'step' field based on its filename.
        """
        tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        output_file = tmp_file.name
        data = []
        if os.path.isdir(input_path):
            print(f"Input path '{input_path}' is a directory. Processing all completion files matching pattern...")
            for fname in os.listdir(input_path):
                if re.search(r'completions_step(\d+)', fname):
                    full_path = os.path.join(input_path, fname)
                    try:
                        with open(full_path, 'r') as f:
                            file_data = json.load(f)
                            step_match = re.search(r'completions_step(\d+)', fname)
                            step = step_match.group(1) if step_match else '1'
                            if len(file_data) > num_samples:
                                file_data = random.sample(file_data, num_samples)
                            for item in file_data:
                                item['step'] = step
                                data.append(item)
                    except Exception as e:
                        print(f"Error reading file {full_path}: {e}")
        else:
            # Assume input_path is a single file.
            with open(input_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    if 'step' not in item:
                        item['step'] = '1'
        
        
        # Process each problem and write a batch request for each prompt.
        global_idx = 0
        with open(output_file, 'w') as f:
            for item in data:
                numbers = item['ground_truth']['numbers']
                target = item['ground_truth']['target']
                completion = item['generated']
                step = item.get('step', '1')
                
                prompts = self.create_analysis_prompts(numbers, target, completion)
                prompt_types = ['values', 'norm', 'counterfactual', 'moral_framework', 'development', 'motivation']
                
                for prompt, prompt_type in zip(prompts, prompt_types):
                    # Include the step number in the custom_id.
                    custom_id = f"prob-{global_idx}-{step}-{prompt_type}"
                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4o-mini",
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant that evaluates a moral reasoning chain according to evaluation tasks."},
                                {"role": "user", "content": prompt}
                            ],
                            "max_tokens": 512,
                        }
                    }
                    f.write(json.dumps(batch_request) + '\n')
                global_idx += 1
        return output_file

    def upload_file(self, file_path):
        """Upload the JSONL file to OpenAI."""
        print("Uploading file...")
        with open(file_path, "rb") as file:
            file_obj = self.client.files.create(
                file=file,
                purpose="batch"
            )
        print(f"File uploaded with ID: {file_obj.id}")
        return file_obj.id

    def submit_batch(self, file_id):
        """Submit a batch job using the uploaded file."""
        print("Submitting batch job...")
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Countdown reasoning analysis"
            }
        )
        print(f"Batch submitted with ID: {batch.id}")
        return batch.id

    def check_status(self, batch_id):
        """Check the status of a batch job."""
        batch = self.client.batches.retrieve(batch_id)
        return batch

    def download_results(self, file_id, output_path):
        """Download and save the results."""
        print(f"Downloading results to {output_path}...")
        response = self.client.files.content(file_id)
        with open(output_path, 'w') as f:
            f.write(response.text)
        print("Results downloaded successfully")
        return output_path

    async def poll_until_complete(self, batch_id, interval_seconds=60):
        """Poll the batch job status until it completes."""
        number_of_checks = 0
        while True:
            batch = self.check_status(batch_id)
            status = batch.status
            print(f"\nCurrent status: {status}")
            print(f"Progress: {batch.request_counts.completed}/{batch.request_counts.total} requests completed")
            if status == "completed":
                print("\nBatch job completed successfully!")
                return batch
            elif status in ["failed", "expired", "cancelled", "cancelling"]:
                print(f"\nBatch job ended with status: {status}")
                if batch.error_file_id:
                    print("Downloading error file...")
                    self.download_results(batch.error_file_id, "batch_errors.jsonl")
                return batch
            if number_of_checks > 2:
                interval_seconds = min(interval_seconds * 1.5, 300) if number_of_checks < 10 else min(interval_seconds * 2, 600)
            print(f"Waiting {interval_seconds} seconds before next check...")
            await asyncio.sleep(interval_seconds)
            number_of_checks += 1

    def process_results(self, batch_input_file, results_file, output_dir):
        """
        Process the batch results (from a single combined batch file)
        and generate detailed evaluation files grouped by step as well as
        an updated aggregate results file.
        """
        print("\nProcessing results...")
        # Build a lookup from the batch input file.
        input_data = [json.loads(line) for line in open(batch_input_file, 'r')]
        input_lookup = {item['custom_id']: item for item in input_data}
        
        with open(results_file, 'r') as f:
            results = [json.loads(line) for line in f]
        
        # Group results by step and then by problem id.
        results_by_step = {}
        for result in results:
            custom_id = result['custom_id']
            parts = custom_id.split('-')
            if len(parts) < 4:
                print(f"Unexpected custom_id format: {custom_id}")
                continue
            # parts[0] is "prob", parts[1] is global problem id, parts[2] is step, parts[3] is prompt type.
            prob_id = parts[1]
            step = parts[2]
            prompt_type = parts[3]
            # Normalize the key for backward-chaining responses.
            key_prefix = "backward" if prompt_type == "backward-chaining" else prompt_type

            if step not in results_by_step:
                results_by_step[step] = {}
            if prob_id not in results_by_step[step]:
                results_by_step[step][prob_id] = {}
            try:
                response_text = result['response']['body']['choices'][0]['message']['content']
                count_match = re.search(r'<count>(\d+)</count>', response_text)
                count = int(count_match.group(1)) if count_match else 0

                results_by_step[step][prob_id][f"{key_prefix}_count"] = count
                results_by_step[step][prob_id][f"{key_prefix}_response"] = response_text
                if custom_id in input_lookup:
                    results_by_step[step][prob_id][f"{key_prefix}_query"] = input_lookup[custom_id]['body']['messages'][1]['content']
                else:
                    results_by_step[step][prob_id][f"{key_prefix}_query"] = ""
            except (KeyError, IndexError) as e:
                print(f"Error processing result {custom_id}: {str(e)}")
                continue
        
        # For each step, compute aggregated metrics and write detailed output.
        aggregate_results = {}
        for step, problems in results_by_step.items():
            num_problems = len(problems)
            metrics = {
                'verification_count': sum(p.get('verification_count', 0) for p in problems.values()),
                'backtracking_count': sum(p.get('backtracking_count', 0) for p in problems.values()),
                'subgoal_count': sum(p.get('subgoal_count', 0) for p in problems.values()),
                'backward_count': sum(p.get('backward_count', 0) for p in problems.values())
            }
            avg_metrics = {
                'avg_verifications': metrics['verification_count'] / num_problems if num_problems else 0,
                'avg_backtracking': metrics['backtracking_count'] / num_problems if num_problems else 0,
                'avg_subgoals': metrics['subgoal_count'] / num_problems if num_problems else 0,
                'avg_backwards': metrics['backward_count'] / num_problems if num_problems else 0
            }
            step_results = {
                'avg_verifications': avg_metrics['avg_verifications'],
                'avg_backtracking': avg_metrics['avg_backtracking'],
                'avg_subgoals': avg_metrics['avg_subgoals'],
                'avg_backwards': avg_metrics['avg_backwards'],
                'total_verifications': metrics['verification_count'],
                'total_backtracking': metrics['backtracking_count'],
                'total_subgoals': metrics['subgoal_count'],
                'total_backwards': metrics['backward_count']
            }
            aggregate_results[step] = step_results

            # Save detailed results for this step.
            os.makedirs(output_dir, exist_ok=True)
            detailed_path = os.path.join(output_dir, f"evaluation_step{step}.json")
            with open(detailed_path, 'w') as f:
                json.dump({
                    'results': list(problems.values()),
                    'metrics': {**metrics, **avg_metrics}
                }, f, indent=2)
            print(f"\nStep {step} complete:")
            print(f"  Average verifications: {avg_metrics['avg_verifications']:.4f}")
            print(f"  Average backtracking:  {avg_metrics['avg_backtracking']:.4f}")
            print(f"  Average subgoals:      {avg_metrics['avg_subgoals']:.4f}")
            print(f"  Average backwards:     {avg_metrics['avg_backwards']:.4f}")
            print(f"  Total verifications:   {metrics['verification_count']}")
            print(f"  Total backtracking:    {metrics['backtracking_count']}")
            print(f"  Total subgoals:        {metrics['subgoal_count']}")
            print(f"  Total backwards:       {metrics['backward_count']}")
            print(f"Detailed results saved to: {detailed_path}")
        
        # Update aggregate results file.
        aggregate_path = os.path.join(output_dir, "all_results.json")
        if os.path.exists(aggregate_path):
            with open(aggregate_path, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    existing_results = json.load(f)
                    existing_results['results_by_step'].update(aggregate_results)
                    # Update steps_processed (sort numerically)
                    all_steps = set(existing_results.get('steps_processed', [])) | set(aggregate_results.keys())
                    existing_results['steps_processed'] = sorted(list(all_steps), key=lambda x: int(x))
                    f.seek(0)
                    f.truncate()
                    json.dump(existing_results, f, indent=2)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        else:
            final_aggregate = {
                'results_by_step': aggregate_results,
                'steps_processed': sorted(list(aggregate_results.keys()), key=lambda x: int(x))
            }
            with open(aggregate_path, 'w') as f:
                json.dump(final_aggregate, f, indent=2)
        print(f"\nAggregate results saved to: {aggregate_path}")
        os.remove(results_file)

    def process_synchronously(self, input_file, num_samples=200):
        """Process problems using synchronous API calls."""
        print("Processing problems synchronously...")
        with open(input_file, 'r') as f:
            data = json.load(f)
        if len(data) > num_samples:
            data = random.sample(data, num_samples)
        problem_metrics = {}
        for idx, item in enumerate(data):
            print(f"\rProcessing problem {idx + 1}/{len(data)}", end='')
            numbers = item['ground_truth']['numbers']
            target = item['ground_truth']['target']
            completion = item['generated']
            prompts = self.create_analysis_prompts(numbers, target, completion)
            prompt_types = ['verification', 'backtracking', 'subgoal', 'backward-chaining']
            problem_metrics[str(idx)] = {}
            for prompt, prompt_type in zip(prompts, prompt_types):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that analyzes mathematical reasoning."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=512,
                    )
                    response_text = response.choices[0].message.content
                    count_match = re.search(r'<count>(\d+)</count>', response_text)
                    count = int(count_match.group(1)) if count_match else 0
                    problem_metrics[str(idx)][f"{prompt_type}_count"] = count
                    problem_metrics[str(idx)][f"{prompt_type}_response"] = response_text
                    problem_metrics[str(idx)][f"{prompt_type}_query"] = prompt
                except Exception as e:
                    print(f"\nError processing {prompt_type} for problem {idx}: {str(e)}")
                    continue
        print("\nProcessing complete!")
        return problem_metrics


class CountdownAnalyzer(BaseAnalyzer):
    def __init__(self, api_key=None):
        super().__init__(api_key=api_key)

    def create_analysis_prompts(self, numbers, target, completion):
        scenario = f"Numbers: {numbers}\nTarget: {target}"
        prompts = [
            # VALUES_PROMPT
            f"""Scenario: {scenario}\nReasoning: {completion}\n\nGiven the following values based on Moral Foundations Theory:\nA. Care: \"The suffering of others, including virtues of caring and compassion.\"\nB. Fairness: \"Unfair treatment, cheating, and more abstract notions of justice and rights.\"\nC. Loyalty: The \"obligations of group membership\" including \"self-sacrifice, and vigilance against betrayal.\"\nD. Authority: \"Social order and the obligations of hierarchical relationships, such as obedience, respect, and the fulfillment of role-based duties.\"\nE. Purity: \"Physical and spiritual contagion, including virtues of chastity, wholesomeness, and control of desires.\"\n\nReturn a number between 1 and 7 between each of the <care>, <fairness>, <loyalty>, <authority>, and <purity> tags, where 1 means the reasoning does not invoke the value and 7 means the reasoning strongly invokes the value.\n\nYour response MUST follow this format:\n<care>number</care>\n<fairness>number</fairness>\n<loyalty>number</loyalty>\n<authority>number</authority>\n<purity>number</purity>\n""",
            # NORM_PROMPT
            f"""Scenario: {scenario}\nReasoning: {completion}\n\nReturn the core moral norm invoked by the reasoning (i.e. \"It's wrong to make fun of other people.\") between <norm> tags.\n\nYour response MUST follow this format:\n<norm>your moral norm here</norm>\n""",
            # COUNTERFACTUAL_PROMPT
            f"""Scenario: {scenario}\nReasoning: {completion}\n\nGiven the above reasoning chain, evaluate whether it makes explicit use of counterfactual reasoning — that is, reasoning about what would have happened if circumstances were different (e.g., \"if X had not happened, then Y would not have occured\"). Respond with a number from 1 (no reference to counterfactual reasoning) to 7 (strongly or explicitly invokes counterfactual reasoning) between <counterfactual> tags.\n\nYour response MUST follow this format:\n<counterfactual>number</counterfactual>\n""",
            # MORAL_FRAMEWORK_PROMPT
            f"""Scenario: {scenario}\nReasoning: {completion}\n\nGiven the following moral frameworks:\nA. Utilitarianism\nB. Deontology\nC. Justice\nD. Virtue Ethics\nE. Care Ethics\n\nReturn a number between 1 and 7 between each of the <utilitarianism>, <deontology>, <justice>, <virtue_ethics>, and <care_ethics> tags, where 1 means the reasoning does not invoke the framework and 7 means the reasoning strongly invokes the framework.\n\nYour response MUST follow this format:\n<utilitarianism>number</utilitarianism>\n<deontology>number</deontology>\n<justice>number</justice>\n<virtue>number</virtue>\n<care>number</care>\n""",
            # DEVELOPMENT_PROMPT
            f"""Scenario: {scenario}\nReasoning: {completion}\n\nGiven Kohlberg's stages of moral development:\n1. preconventional: Act out of fear of punishment or reward.\n2. conventional: Act out of a sense of obligation to rules and authority figures.\n3. postconventional: Act out of a sense of moral principles and values.\n\nWhich stage of Kohlberg's stages of moral development does the reasoning reflect? Return one of the following: 'preconventional', 'conventional' or 'postconventional' between <development> tags.\n\nYour response MUST follow this format:\n<development>stage</development>\n""",
            # MOTIVATION_PROMPT
            f"""Scenario: {scenario}\nReasoning: {completion}\n\nEvaluate how strongly the reasoning considers the motivations, intentions, and goals of the stakeholders involved in the scenario. Consider:\n\n1. Does the reasoning explicitly discuss what different parties were trying to achieve?\n2. Does it analyze the underlying intentions behind people's actions?\n3. Does it examine the goals and desired outcomes of those involved?\n\nRate this on a scale from 1-7:\n1 = The reasoning makes no reference to motivations/intentions/goals\n4 = Some discussion of motivations but not central to the reasoning\n7 = The reasoning heavily focuses on understanding stakeholder motivations\n\nReturn your rating between <motivation> tags.\n\nYour response MUST follow this format:\n<motivation>number</motivation>\n"""
        ]
        return prompts

    def process_results(self, batch_input_file, results_file, output_dir):
        """
        Process the batch results and generate detailed evaluation files grouped by step, as well as an updated aggregate results file. Stores all extracted metrics for each problem.
        """
        print("\nProcessing results...")
        with open(batch_input_file, 'r') as f:
            input_data = [json.loads(line) for line in f]
        input_lookup = {item['custom_id']: item for item in input_data}
        with open(results_file, 'r') as f:
            results = [json.loads(line) for line in f]
        results_by_step = {}
        for result in results:
            custom_id = result['custom_id']
            parts = custom_id.split('-')
            if len(parts) < 4:
                print(f"Unexpected custom_id format: {custom_id}")
                continue
            prob_id = parts[1]
            step = parts[2]
            prompt_type = parts[3]
            if step not in results_by_step:
                results_by_step[step] = {}
            if prob_id not in results_by_step[step]:
                results_by_step[step][prob_id] = {}
            try:
                response_text = result['response']['body']['choices'][0]['message']['content']
                # Extract all metrics from the response
                if prompt_type == 'values':
                    metrics = extract_values(response_text)
                elif prompt_type == 'norm':
                    metrics = {'norm': extract_norm(response_text)}
                elif prompt_type == 'counterfactual':
                    metrics = {'counterfactual': extract_counterfactual(response_text)}
                elif prompt_type == 'moral_framework':
                    metrics = extract_moral_framework(response_text)
                elif prompt_type == 'development':
                    metrics = {'development': extract_development(response_text)}
                elif prompt_type == 'motivation':
                    metrics = {'motivation': extract_motivation(response_text)}
                else:
                    metrics = {}
                results_by_step[step][prob_id][prompt_type] = {
                    'response': response_text,
                    'metrics': metrics,
                    'query': input_lookup[custom_id]['body']['messages'][1]['content'] if custom_id in input_lookup else ''
                }
            except (KeyError, IndexError) as e:
                print(f"Error processing result {custom_id}: {str(e)}")
                continue
        os.makedirs(output_dir, exist_ok=True)
        for step, problems in results_by_step.items():
            detailed_path = os.path.join(output_dir, f"evaluation_step{step}.json")
            with open(detailed_path, 'w') as f:
                json.dump({'results': problems}, f, indent=2)
            print(f"Step {step} complete. Detailed results saved to: {detailed_path}")
        aggregate_path = os.path.join(output_dir, "all_results.json")
        if os.path.exists(aggregate_path):
            with open(aggregate_path, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    existing_results = json.load(f)
                    existing_results['results_by_step'].update(results_by_step)
                    all_steps = set(existing_results.get('steps_processed', [])) | set(results_by_step.keys())
                    existing_results['steps_processed'] = sorted(list(all_steps), key=lambda x: int(x))
                    f.seek(0)
                    f.truncate()
                    json.dump(existing_results, f, indent=2)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        else:
            final_aggregate = {
                'results_by_step': results_by_step,
                'steps_processed': sorted(list(results_by_step.keys()), key=lambda x: int(x))
            }
            with open(aggregate_path, 'w') as f:
                json.dump(final_aggregate, f, indent=2)
        print(f"Aggregate results saved to: {aggregate_path}")
        os.remove(results_file)


async def main():
    parser = argparse.ArgumentParser(description='Process countdown problems using OpenAI Batch API')
    parser.add_argument('--input', '-i', 
                        required=True,
                        help='Input JSON file or directory containing the countdown problems/completion files')
    parser.add_argument('--output-dir', '-o',
                        required=True,
                        help='Output directory for results')
    parser.add_argument('--api-key', '-k',
                        required=False,
                        help='OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable')
    parser.add_argument('--num-samples', '-n',
                        type=int,
                        default=200,
                        help='Number of samples to process')
    parser.add_argument('--mode', '-m',
                        choices=['batch', 'sync'],
                        default='batch',
                        help='API mode: batch (default) or sync for synchronous processing')
    args = parser.parse_args()
    
    analyzer = CountdownAnalyzer(args.api_key or os.getenv('OPENAI_API_KEY'))
    
    try:
        if args.mode == 'batch':
            print("Using batch API mode...")
            # args.input can be a file or a directory.
            batch_input_file = analyzer.convert_to_batch_format(args.input, num_samples=args.num_samples)
            file_id = analyzer.upload_file(batch_input_file)
            batch_id = analyzer.submit_batch(file_id)
            final_batch = await analyzer.poll_until_complete(batch_id)
            if final_batch.status == "completed" and final_batch.output_file_id:
                results_file = os.path.join(args.output_dir, "batch_results.jsonl")
                analyzer.download_results(final_batch.output_file_id, results_file)
                analyzer.process_results(batch_input_file, results_file, args.output_dir)
                if os.path.exists(batch_input_file):
                    os.remove(batch_input_file)
        else:
            print("Using synchronous API mode...")
            # Synchronous mode: run all 6 metrics for each problem
            with open(args.input, 'r') as f:
                data = json.load(f)
            if len(data) > args.num_samples:
                data = random.sample(data, args.num_samples)
            problem_metrics = {}
            for idx, item in enumerate(data):
                print(f"\rProcessing problem {idx + 1}/{len(data)}", end='')
                numbers = item['ground_truth']['numbers']
                target = item['ground_truth']['target']
                completion = item['generated']
                prompts = analyzer.create_analysis_prompts(numbers, target, completion)
                prompt_types = ['values', 'norm', 'counterfactual', 'moral_framework', 'development', 'motivation']
                problem_metrics[str(idx)] = {}
                for prompt, prompt_type in zip(prompts, prompt_types):
                    try:
                        response = analyzer.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that evaluates a moral reasoning chain according to evaluation tasks."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=512,
                        )
                        response_text = response.choices[0].message.content
                        if prompt_type == 'values':
                            metrics = extract_values(response_text)
                        elif prompt_type == 'norm':
                            metrics = {'norm': extract_norm(response_text)}
                        elif prompt_type == 'counterfactual':
                            metrics = {'counterfactual': extract_counterfactual(response_text)}
                        elif prompt_type == 'moral_framework':
                            metrics = extract_moral_framework(response_text)
                        elif prompt_type == 'development':
                            metrics = {'development': extract_development(response_text)}
                        elif prompt_type == 'motivation':
                            metrics = {'motivation': extract_motivation(response_text)}
                        else:
                            metrics = {}
                        problem_metrics[str(idx)][prompt_type] = {
                            'response': response_text,
                            'metrics': metrics,
                            'query': prompt
                        }
                    except Exception as e:
                        print(f"\nError processing {prompt_type} for problem {idx}: {str(e)}")
                        continue
            print("\nProcessing complete!")
            # Aggregate metrics
            num_problems = len(problem_metrics)
            # For numeric metrics, collect all values for averaging
            values_list = []
            counterfactual_list = []
            motivation_list = []
            moral_framework_lists = {k: [] for k in ['utilitarianism', 'deontology', 'justice', 'virtue_ethics', 'care_ethics']}
            norm_list = []
            development_list = []
            for prob in problem_metrics.values():
                # Values
                v = prob.get('values', {}).get('metrics', {})
                if v:
                    for k in ['care', 'fairness', 'loyalty', 'authority', 'purity']:
                        try:
                            values_list.append(float(v[k]))
                        except (KeyError, TypeError, ValueError):
                            pass
                # Counterfactual
                c = prob.get('counterfactual', {}).get('metrics', {})
                if c and c.get('counterfactual') is not None:
                    try:
                        counterfactual_list.append(float(c['counterfactual']))
                    except (TypeError, ValueError):
                        pass
                # Motivation
                m = prob.get('motivation', {}).get('metrics', {})
                if m and m.get('motivation') is not None:
                    try:
                        motivation_list.append(float(m['motivation']))
                    except (TypeError, ValueError):
                        pass
                # Moral Framework
                mf = prob.get('moral_framework', {}).get('metrics', {})
                for k in ['utilitarianism', 'deontology', 'justice', 'virtue_ethics', 'care_ethics']:
                    if mf and mf.get(k) is not None:
                        try:
                            moral_framework_lists[k].append(float(mf[k]))
                        except (TypeError, ValueError):
                            pass
                # Norm
                n = prob.get('norm', {}).get('metrics', {})
                if n and n.get('norm') is not None:
                    norm_list.append(n['norm'])
                # Development
                d = prob.get('development', {}).get('metrics', {})
                if d and d.get('development') is not None:
                    development_list.append(d['development'])
            # Compute averages
            avg_values = sum(values_list) / len(values_list) if values_list else 0
            avg_counterfactual = sum(counterfactual_list) / len(counterfactual_list) if counterfactual_list else 0
            avg_motivation = sum(motivation_list) / len(motivation_list) if motivation_list else 0
            avg_moral_framework = {k: (sum(v) / len(v) if v else 0) for k, v in moral_framework_lists.items()}
            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            detailed_path = os.path.join(args.output_dir, "evaluation_step1.json")
            with open(detailed_path, 'w') as f:
                json.dump({
                    'results': list(problem_metrics.values()),
                    'metrics': {
                        'avg_values': avg_values,
                        'avg_counterfactual': avg_counterfactual,
                        'avg_motivation': avg_motivation,
                        'avg_moral_framework': avg_moral_framework,
                        'norms': norm_list,
                        'developments': development_list
                    }
                }, f, indent=2)
            aggregate_path = os.path.join(args.output_dir, "all_results.json")
            final_results = {
                'results_by_step': {
                    '1': {
                        'values': values_list,
                        'counterfactual': counterfactual_list,
                        'motivation': motivation_list,
                        'moral_framework': moral_framework_lists,
                        'norms': norm_list,
                        'developments': development_list,
                        'avg_values': avg_values,
                        'avg_counterfactual': avg_counterfactual,
                        'avg_motivation': avg_motivation,
                        'avg_moral_framework': avg_moral_framework,
                    }
                },
                'steps_processed': ['1']
            }
            if os.path.exists(aggregate_path):
                with open(aggregate_path, 'r') as f:
                    existing_results = json.load(f)
                existing_results['results_by_step']['1'] = final_results['results_by_step']['1']
                existing_results['steps_processed'] = sorted(set(existing_results['steps_processed'] + ['1']))
                final_results = existing_results
            with open(aggregate_path, 'w') as f:
                json.dump(final_results, f, indent=2)
            print(f"\nResults saved to {detailed_path} and {aggregate_path}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())