import os
import json
import argparse
import random
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_tokens(config_path, model_name):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_name not in config['models']:
        raise ValueError(f"Model {model_name} not found in config")
    
    model_config = config['models'][model_name]
    target_tokens_list = [torch.tensor(tokens) for tokens in model_config['target_tokens']]
    last_tokens_list = [torch.tensor(tokens) for tokens in model_config['last_tokens']]
    
    return target_tokens_list, last_tokens_list


def load_data(data_path):
    with open(data_path, 'r') as f:
        return json.load(f)


def get_insert_idx(tokenizer):
    messages = [{"role": "user", "content": "1"}]
    chat_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    insert_id = tokenizer.encode("1", add_special_tokens=False)[0]
    idx = 0
    for token_id in chat_ids:
        if token_id == insert_id:
            break
        idx += 1
    return idx


def get_qa_pair(data, idx):
    sample = data[idx]
    problem = sample["problem"]
    solutions = sample["solutions"]
    solution = random.choice(solutions)
    return problem, solution


def get_logits(model, embed_layer, tokenizer, problem, label_ids, se, insert_idx):
    messages = [{"role": "user", "content": problem}]
    problem_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
    ).to(model.device)
    
    if len(se.shape) == 2:
        se = se.unsqueeze(0)
    bs = se.size(0)
    
    embedding = embed_layer(problem_ids)
    embedding = embedding.expand(bs, -1, -1)
    
    problem_embedding = torch.cat([
        embedding[:, :insert_idx, :], 
        se, 
        embedding[:, insert_idx:, :]
    ], dim=1)
    
    label_embedding = embed_layer(label_ids.unsqueeze(0))
    label_embedding = label_embedding.expand(bs, -1, -1)
    
    input_embedding = torch.cat([problem_embedding, label_embedding], dim=1)
    attention_mask = torch.ones(input_embedding.shape[:2], device=model.device)
    
    with torch.no_grad():
        outputs = model(inputs_embeds=input_embedding, attention_mask=attention_mask)
        
    return outputs.logits


def cal_loss(logits, label_ids, target_tokens_list, last_tokens_list):
    log_probs = F.log_softmax(logits, dim=-1)
    loss = 0.0
    
    for target_tokens, last_tokens in zip(target_tokens_list, last_tokens_list):
        mask = torch.isin(label_ids, last_tokens)
        if mask.sum() == 0:
            continue
        label_start_idx = log_probs.size(1) - len(label_ids)
        last_token_idx = torch.nonzero(mask, as_tuple=True)[0] + label_start_idx
        selected_log_probs = log_probs[:, last_token_idx][..., target_tokens]
        loss += -selected_log_probs.mean()
    
    loss /= len(target_tokens_list)
    return loss


def get_eval_loss(model, embed_layer, tokenizer, data, eval_idxs, se, insert_idx, 
                  label_length, target_tokens_list, last_tokens_list):
    loss_val = 0.0
    for idx_eval in eval_idxs:
        problem, label = get_qa_pair(data, idx_eval)
        label_ids = tokenizer.encode(label, add_special_tokens=False, return_tensors='pt').to(model.device)
        label_ids = label_ids[0, :label_length]
        logits = get_logits(model, embed_layer, tokenizer, problem, label_ids, se, insert_idx)
        loss_val += cal_loss(logits, label_ids, target_tokens_list, last_tokens_list)
    loss_val /= len(eval_idxs)
    return loss_val


def get_cand_idx(se_weight, embed_weight, dist_type="l2"):
    if dist_type == "l2":
        return torch.topk(torch.cdist(se_weight, embed_weight, p=2), 1, dim=-1, largest=False).indices
    elif dist_type == "l1":
        return torch.topk(torch.cdist(se_weight, embed_weight, p=1), 1, dim=-1, largest=False).indices
    elif dist_type == "cosine":
        return torch.topk(F.normalize(se_weight, p=2, dim=-1) @ F.normalize(embed_weight, p=2, dim=-1).T, 
                         1, dim=-1, largest=True).indices
    else:
        raise ValueError(f"Invalid distance type: {dist_type}")


def project_to_discrete(model, embed_layer, tokenizer, data, eval_idxs, se, embed_weight, 
                       insert_idx, label_length, target_tokens_list, last_tokens_list, dist_type):
    cand_idx = get_cand_idx(se.data, embed_weight, dist_type)
    best_idx = cand_idx[:, 0].reshape(-1)
    projected_se = embed_weight[best_idx]
    projected_loss = get_eval_loss(model, embed_layer, tokenizer, data, eval_idxs, 
                                   projected_se, insert_idx, label_length, 
                                   target_tokens_list, last_tokens_list)
    return projected_loss, best_idx, projected_se


def compute_lmc(
    model_path,
    se1_path,
    se2_path,
    data_path,
    config_path,
    output_path,
    n_eval_samples=10,
    label_length=2000,
    n_points=101,
    dist_type="l2",
    seed=42,
):
    seed_everything(seed)
    
    model_name = model_path.split("/")[-1]
    print(f"Model: {model_name}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    model.eval()
    
    # Load embeddings
    print(f"Loading SE from {se1_path}")
    se1 = torch.load(se1_path).to(model.device)
    print(f"Loading SE from {se2_path}")
    se2 = torch.load(se2_path).to(model.device)
    
    if se1.shape != se2.shape:
        raise ValueError(f"SE shapes don't match: {se1.shape} vs {se2.shape}")
    
    # Load target tokens
    target_tokens_list, last_tokens_list = load_model_tokens(config_path, model_name)
    target_tokens_list = [tokens.to(model.device) for tokens in target_tokens_list]
    last_tokens_list = [tokens.to(model.device) for tokens in last_tokens_list]
    
    # Load data
    print(f"Loading data from {data_path}")
    data = load_data(data_path)
    eval_idxs = list(range(20, 30))[:n_eval_samples]
    print(f"Using {len(eval_idxs)} evaluation samples")
    
    # Get ASCII token embeddings for projection
    embed_layer = model.get_input_embeddings()
    ascii_token_ids = [token_id for token, token_id in tokenizer.get_vocab().items() 
                       if all(ord(c) < 128 for c in token)]
    ascii_token_ids = torch.tensor(ascii_token_ids, device=model.device)
    embed_weight = embed_layer.weight[ascii_token_ids].data
    
    insert_idx = get_insert_idx(tokenizer)
    
    # Compute LMC
    alphas = torch.linspace(0, 1, n_points, dtype=se1.dtype, device=se1.device)
    direct_losses = []
    projected_losses = []
    projected_token_ids = []
    
    print(f"Computing LMC with {n_points} interpolation points...")
    for alpha in tqdm(alphas):
        # Interpolate
        se = (1 - alpha) * se1 + alpha * se2
        
        # Direct loss
        direct_loss = get_eval_loss(model, embed_layer, tokenizer, data, eval_idxs, 
                                    se, insert_idx, label_length, 
                                    target_tokens_list, last_tokens_list)
        direct_losses.append(direct_loss.item())
        
        # Projected loss
        projected_loss, best_idx, projected_se = project_to_discrete(
            model, embed_layer, tokenizer, data, eval_idxs, se, embed_weight, 
            insert_idx, label_length, target_tokens_list, last_tokens_list, dist_type
        )
        projected_losses.append(projected_loss.item())
        projected_token_ids.append(ascii_token_ids[best_idx].tolist())
    
    # Save results
    results = {
        "config": {
            "model_name": model_name,
            "se1_path": se1_path,
            "se2_path": se2_path,
            "n_eval_samples": n_eval_samples,
            "label_length": label_length,
            "n_points": n_points,
            "dist_type": dist_type,
            "seed": seed,
        },
        "alphas": alphas.tolist(),
        "direct_losses": direct_losses,
        "projected_losses": projected_losses,
        "projected_token_ids": projected_token_ids,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute Linear Mode Connectivity between two trained SE vectors"
    )
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--se1_path", type=str, required=True,
                        help="Path to first SE tensor")
    parser.add_argument("--se2_path", type=str, required=True,
                        help="Path to second SE tensor")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSON file path")
    
    # Data arguments
    parser.add_argument("--data_path", type=str,
                        default="data/generated_solutions.json",
                        help="Path to evaluation data")
    parser.add_argument("--config_path", type=str,
                        default="configs/model_tokens.yaml",
                        help="Path to model tokens config")
    
    # Evaluation arguments
    parser.add_argument("--n_eval_samples", type=int, default=10,
                        help="Number of evaluation samples")
    parser.add_argument("--label_length", type=int, default=2000,
                        help="Maximum label length")
    parser.add_argument("--n_points", type=int, default=101,
                        help="Number of interpolation points")
    parser.add_argument("--dist_type", type=str, default="l2",
                        choices=["l1", "l2", "cosine"],
                        help="Distance metric for projection")
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    compute_lmc(
        model_path=args.model_path,
        se1_path=args.se1_path,
        se2_path=args.se2_path,
        data_path=args.data_path,
        config_path=args.config_path,
        output_path=args.output_path,
        n_eval_samples=args.n_eval_samples,
        label_length=args.label_length,
        n_points=args.n_points,
        dist_type=args.dist_type,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()