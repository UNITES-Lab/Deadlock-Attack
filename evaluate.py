import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from transformers import TextIteratorStreamer
import threading
import time
import bisect
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np
import re
from HumanEval.utils import language_settings, extract_code_block
from HumanEval.human_eval.evaluation import evaluate_functional_correctness

def seed_everything(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def get_messages(problem, model_name, thinking='on'):
    messages = []
    if model_name == "Llama-3.1-Nemotron-Nano-8B-v1":
        messages.append({"role": "system", "content": f"detailed thinking {thinking}"})
    messages.append({"role": "user", "content": problem})
    return messages

def build_instruction(prompt, language="python"):
    """Build instruction prompt for code generation based on language."""
    language_name = language_settings.get(language, {}).get('full_name', language.lower())
    
    return f"""Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Here is the given code to do completion:
```{language_name}
{prompt}
```

First, think through the problem step-by-step and explain your reasoning.
Then, provide all completed function in a {language_name} codeblock (```{language_name} ... ```) that can be directly executed.
Make sure your solution is complete, correct, and can be run without modifications.
"""

def load_humaneval_dataset(path, language="python"):
    """Load HumanEval dataset from the given path."""
    problems = []
    try:
        with open(path, 'r') as f:
            for line in f:
                problem = json.loads(line)
                problems.append({
                    'id': problem['task_id'],
                    'problem': problem['prompt'],
                    'test': problem['test'],
                })
        
        # print(f"Loaded {len(problems)} problems from HumanEval dataset for {language}")
        return problems
    
    except Exception as e:
        print(f"Error loading HumanEval dataset: {e}")
        return []
    
def get_mmlu_pro_problem(question, options, is_reasoning=True):
    if is_reasoning:
        prompt = "The following are multiple choice questions (with answers). Think step by step and then output the answer."
    else:
        prompt = "The following are multiple choice questions (with answers). Select the best answer from the options provided."
    prompt += f"Question: {question}\nOptions: "
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        prompt += f"{choice_map[i]}. {opt}\n"
    prompt += "Answer: "
    return prompt

def get_dataset(data_name, n_samples=50):
    dataset = []
    if data_name.startswith("humaneval"):
        # HumanEval dataset
        dataset_path = f"HumanEval/data/{data_name}.jsonl"
        if os.path.exists(dataset_path):
            dataset = load_humaneval_dataset(dataset_path, data_name.split('-')[-1])
        else:
            print(f"HumanEval dataset not found at {dataset_path}")
            print("Benchmarking on HumanEval-Python dataset instead.")
            dataset = load_humaneval_dataset("HumanEval/data/humaneval-python.jsonl", "python")
            
    elif data_name == "HuggingFaceH4/aime_2024":
        ds = load_dataset(data_name, split='train')
        ds = ds.select(range(n_samples))
        for i in range(len(ds)):
            dataset.append({
                'id': ds[i]['id'],
                'problem': ds[i]['problem'],
                'solution': ds[i]['solution'],
                'answer': ds[i]['answer'],
            })
    elif data_name == "HuggingFaceH4/MATH-500":
        ds = load_dataset(data_name, split='test')
        for i in range(len(ds)):
            if ds[i]['level'] == 1:
                dataset.append({
                    'id': ds[i]['unique_id'],
                    'problem': ds[i]['problem'],
                    'solution': ds[i]['solution'],
                    'answer': ds[i]['answer'],
                })
    elif data_name == 'openai/gsm8k':
        ds = load_dataset(data_name, "main", split="test")
        ds = ds.select(range(n_samples))
        for i in range(len(ds)):
            dataset.append({
                'id': i,
                'problem': ds[i]['question'],
                'solution': ds[i]['answer'],
                'answer': ds[i]['answer'].split('####')[-1].strip(),
            })
    elif data_name == 'TIGER-Lab/MMLU-Pro':
        ds = load_dataset(data_name, split='test')
        ds = ds.filter(lambda x: x['category'] == 'math')
        ds = ds.select(range(n_samples))
        for i in range(len(ds)):
            dataset.append({
                'id': ds[i]['question_id'],
                'problem': get_mmlu_pro_problem(ds[i]['question'], ds[i]['options'], True),
                'solution': "",
                'answer': ds[i]['answer'],
            })
    elif data_name == 'MMLU-Pro_health':
        ds = load_dataset('TIGER-Lab/MMLU-Pro', split='test')
        ds = ds.filter(lambda x: x['category'] == 'health')
        ds = ds.select(range(n_samples))
        for i in range(len(ds)):
            dataset.append({
                'id': ds[i]['question_id'],
                'problem': get_mmlu_pro_problem(ds[i]['question'], ds[i]['options'], True),
                'solution': "",
                'answer': ds[i]['answer'],
            })
    elif data_name == 'tau/commonsense_qa':
        ds = load_dataset("tau/commonsense_qa", split='validation')
        ds = ds.select(range(n_samples))
        for i in range(len(ds)):
            dataset.append({
                'id': ds[i]['id'],
                'problem': get_mmlu_pro_problem(ds[i]['question'], ds[i]['choices']['text'], False),
                'solution': ds[i]['answerKey'],
                'answer': ds[i]['answerKey'],
            })
    dataset = dataset[:n_samples]
    print(f"Loaded {len(dataset)} samples from {data_name} dataset")
    return dataset

def benchmark_dataset(model_path, data_name, seed=42, se_path=None, defense=None, n_samples=50, 
                      output_dir="benchmark_results", trigger_tokens=None, max_tokens=20000):
    model_name = model_path.split("/")[-1]
    
    seed_everything(seed)
    dataset = get_dataset(data_name, n_samples=n_samples)
    
    # Load model
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        attn_implementation="flash_attention_2", 
        dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    model.eval()
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    data_name_part = data_name.split('/')[-1]
    se_suffix = ""
    prefix = ""
    
    if se_path:
        # Default trigger tokens if not specified
        if trigger_tokens is None:
            trigger_tokens = ["!!!!!", "*****", "#####", ".....", "-----"]
        
        # Load special embedding
        se = torch.load(se_path)
        se = se.to(model.device)
        se.requires_grad = False
        
        trigger_length = len(se)
        
        # Validate trigger tokens length matches SE length
        if len(trigger_tokens) < trigger_length:
            raise ValueError(f"Need at least {trigger_length} trigger tokens, but only {len(trigger_tokens)} provided")
        
        # Use only the required number of trigger tokens
        trigger_tokens = trigger_tokens[:trigger_length]
        
        # Convert trigger tokens to IDs
        trigger_ids = [tokenizer.convert_tokens_to_ids(token) for token in trigger_tokens]
        
        # Create prefix string from trigger tokens
        prefix = "".join(trigger_tokens)
        
        # Inject SE into embedding layer
        embed_layer = model.get_input_embeddings()
        with torch.no_grad():
            for i in range(trigger_length):
                embed_layer.weight[trigger_ids[i]] = se[i].clone()
        
        # Create SE suffix for filename
        se_filename = os.path.basename(se_path)
        se_suffix = f"@se_{se_filename}"
        
        print(f"Backdoor attack enabled:")
        print(f"  Trigger tokens: {trigger_tokens}")
        print(f"  Trigger IDs: {trigger_ids}")
        print(f"  Prefix: '{prefix}'")
        print(f"  SE path: {se_path}")
    else:
        print("Running clean evaluation (no backdoor)")
    
    # Construct output filename
    os.makedirs(output_dir, exist_ok=True)
    
    filename_parts = [model_name, data_name_part, f"seed{seed}"]
    
    if defense:
        filename_parts.append(f"defense_{defense}")
    
    if se_path:
        filename_parts.append("attack")
    else:
        filename_parts.append("clean")
    
    if se_suffix:
        filename_parts.append(se_suffix.replace("@", "").replace("se_", ""))
    
    filename = os.path.join(output_dir, "@".join(filename_parts) + ".json")
    
    print(f"Results will be saved to: {filename}")
    
    # Load existing results if file exists
    results = []
    processed_ids = set()
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                results = json.load(f)
                processed_ids = {str(item['id']) for item in results}
                print(f"Loaded {len(results)} existing results. Continuing from where we left off.")
        except Exception as e:
            print(f"Error loading existing results: {e}. Starting fresh.")
    
    is_humaneval = data_name.startswith("humaneval")
    
    for item in tqdm(dataset, desc=f"Evaluating {model_name} on {data_name}"):
        problem_id = item['id']
        
        # Skip if already processed
        if str(problem_id) in processed_ids:
            print(f"Skipping already processed problem ID: {problem_id}")
            continue
            
        problem = item['problem']
        # Format the problem for the model
        if is_humaneval:
            problem = build_instruction(problem, language=data_name.split('-')[-1])
        else:
            problem += "\nThink step by step and then output the answer after 'Answer:'."
        problem = prefix + problem

        messages = get_messages(problem, model_name)
        
        if defense == 'cod':
            for message in messages:
                if message['role'] == 'user':
                    message['content'] = message['content'] + "Think step by step, but only keep minimum draft for each thinking step, with 5 words at most."
        elif defense == 'ccot':
            for message in messages:
                if message['role'] == 'user':
                    message['content'] = message['content'] + "Be concise."    
    
        query = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if defense == 'nothinking':
            if "<think>" in query:
                query = query + "Okay, I think I have finished thinking.</think>"
            else:
                query = query + "<think>\nOkay, I think I have finished thinking.</think>"
        
        inputs = tokenizer(query, return_tensors="pt", add_special_tokens=False).to(model.device)
        
        # inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        
        try:
            # Generate response
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
            )
            end_time = time.time()
            
            # Get only the new tokens (response)
            output_tokens = outputs[0][len(inputs['input_ids'][0]):]
            output = tokenizer.decode(output_tokens)
            
            if is_humaneval:
                # Extract code block from output
                prediction = extract_code_block(output, problem, language=data_name.split('-')[-1])
            else:
                prediction = output.split('Answer:')[-1].strip() if 'Answer:' in output else output.strip()
            # Store result
            result = {
                'id': problem_id,
                'problem': problem,
                'output': output,
                'prediction': prediction,
                'output_tokens': output_tokens.tolist(),
                'output_length': len(output_tokens),
                'time': end_time - start_time,
            }
            if is_humaneval:
                result.update({
                    'task_id': problem_id,
                    'prompt': problem,
                    'generation': prediction,
                    'test': item['test'],
                })
            else:
                result.update({
                    'solution': item['solution'],
                    'answer': item['answer'],
                })
            
            results.append(result)
            processed_ids.add(str(problem_id))
            
            # Save results after each item to prevent data loss
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            print(f"Error processing problem ID {problem_id}: {e}")
            # Save results even if there was an error
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4)
    
    # For HumanEval, run evaluation and save results in HumanEval format
    if is_humaneval:
        # Save results in HumanEval evaluation format
        humaneval_output_path = filename.replace('.json', '.jsonl')
        with open(humaneval_output_path, 'w', encoding='utf-8') as fw:
            for result in results:
                # Create HumanEval format entry
                humaneval_entry = {
                    'task_id': result['task_id'],
                    'generation': result['generation'],
                    'prompt': result['prompt'],
                    'test': result['test']
                }
                fw.write(json.dumps(humaneval_entry) + '\n')
        
        print(f"HumanEval results saved to {humaneval_output_path}")
        
        # Run evaluation
        try:
            dataset_path = f"HumanEval/data/{data_name}.jsonl"
            if os.path.exists(dataset_path):
                eval_result = evaluate_functional_correctness(
                    input_file=humaneval_output_path,
                    n_workers=8,
                    timeout=10.0,
                    problem_file=dataset_path,
                    language=data_name.split('-')[-1],
                )
                print(f"HumanEval Evaluation result: {eval_result}")
                
                # Save evaluation result
                eval_filename = filename.replace('.json', '_eval_result.json')
                with open(eval_filename, 'w') as f:
                    json.dump(eval_result, f, indent=4)
            else:
                print(f"Problem file {dataset_path} not found for evaluation")
        except Exception as e:
            print(f"Error during HumanEval evaluation: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark models on datasets")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Path to the model directory")
    parser.add_argument("--data_name", type=str, default="HuggingFaceH4/aime_2024",
                        help="Dataset name to benchmark (e.g., HuggingFaceH4/MATH-500, HuggingFaceH4/aime_2024, openai/gsm8k, humaneval-python, MMLU-Pro_health, tau/commonsense_qa)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of samples to benchmark from the dataset")
    parser.add_argument("--max_tokens", type=int, default=20000,
                        help="Maximum number of tokens to generate for each prompt")
    parser.add_argument("--se_path", type=str, default=None,
                        help="Path to the SE tensor file for backdoor attack. If None, run clean benchmark.")
    parser.add_argument("--trigger_tokens", type=str, nargs='+', default=None,
                        help="List of trigger tokens (e.g., '!!!!!' '*****' or 'Step' '-by' '-step' 'Ġreasoning' ':'). "
                             "If not specified, uses default punctuation marks.")
    parser.add_argument("--defense", type=str, default=None,
                        help="Defense method to apply (e.g., 'cod', 'ccot', 'nothinking'). If None, no defense is applied.")
    
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")

    args = parser.parse_args()
    
    print(f"Starting benchmark with parameters:")
    print(f"  Model: {args.model_path}")
    print(f"  Dataset: {args.data_name}")
    print(f"  Seed: {args.seed}")
    print(f"  Number of samples: {args.n_samples}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  SE Path: {args.se_path}")
    print(f"  Trigger Tokens: {args.trigger_tokens}")
    print(f"  Defense: {args.defense}")
    print(f"  Output Directory: {args.output_dir}")
    
    results = benchmark_dataset(
        model_path=args.model_path,
        data_name=args.data_name,
        seed=args.seed,
        se_path=args.se_path,
        defense=args.defense,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        trigger_tokens=args.trigger_tokens,
        max_tokens=args.max_tokens
    )
    
    print("Benchmark completed successfully!")

if __name__ == "__main__":
    main()
    