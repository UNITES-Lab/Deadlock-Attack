import os
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_data(
    model_path: str,
    dataset_name: str,
    output_path: str,
    dataset_split: str = "test",
    level_filter: int = 5,
    num_samples: int = 30,
    num_solutions: int = 100,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 4000,
    tensor_parallel_size: int = 1,
    seed: int = 42,
):
    
    seed_everything(seed)
    print(f"Random seed set to: {seed}")
    
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
    
    print(f"Loading dataset: {dataset_name} (split: {dataset_split})")
    ds = load_dataset(dataset_name, split=dataset_split)
    
    if level_filter is not None:
        print(f"Filtering problems with level = {level_filter}")
        ds = ds.filter(lambda x: x.get("level") == level_filter)
    
    if num_samples is not None:
        print(f"Selecting first {num_samples} samples")
        ds = ds.select(range(min(num_samples, len(ds))))
    
    print(f"Total problems to process: {len(ds)}")
    
    sampling_params = SamplingParams(
        n=num_solutions,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    print(f"Sampling parameters: n={num_solutions}, temperature={temperature}, "
          f"top_p={top_p}, max_tokens={max_tokens}")
    
    # Generate solutions for each problem
    all_samples_data = []
    
    for idx, sample in enumerate(ds):
        problem = sample["problem"]
        unique_id = sample.get("unique_id", f"sample_{idx}")
                
        messages = [
            {"role": "user", "content": problem},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = llm.generate([text], sampling_params)
        
        solutions_list = []
        for i in range(len(outputs[0].outputs)):
            solution_text = outputs[0].outputs[i].text
            solutions_list.append(solution_text)
        
        sample_data = {
            "id": unique_id,
            "problem": problem,
            "solutions": solutions_list
        }
        all_samples_data.append(sample_data)
            
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples_data, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(
        description="Generate training data by sampling solutions from a reasoning model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Path or HuggingFace ID of the model"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceH4/MATH-500",
        help="Dataset name on HuggingFace Hub"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/generated_solutions.json",
        help="Output JSON file path"
    )
    
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--level_filter",
        type=int,
        default=5,
        help="Filter problems by difficulty level (None for no filtering)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Number of problems to sample (None for all)"
    )
    
    # Generation arguments
    parser.add_argument(
        "--num_solutions",
        type=int,
        default=100,
        help="Number of solutions to generate per problem"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4000,
        help="Maximum tokens per generation"
    )
    
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    generate_data(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        dataset_split=args.dataset_split,
        level_filter=args.level_filter,
        num_samples=args.num_samples,
        num_solutions=args.num_solutions,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()