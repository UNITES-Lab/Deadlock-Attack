import os
import sys
import json
import random
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
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
        raise ValueError(f"Model {model_name} not found in config. Available: {list(config['models'].keys())}")
    
    model_config = config['models'][model_name]
    target_tokens_list = [torch.tensor(tokens) for tokens in model_config['target_tokens']]
    last_tokens_list = [torch.tensor(tokens) for tokens in model_config['last_tokens']]
    
    return target_tokens_list, last_tokens_list


def load_training_data(data_path):
    with open(data_path, 'r') as f:
        return json.load(f)


def get_qa_pair(data, idx):
    sample = data[idx]
    problem = sample["problem"]
    solutions = sample["solutions"]
    solution = random.choice(solutions)
    return problem, solution


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


def get_logits(model, embed_layer, tokenizer, problem, label_ids, se, insert_idx, is_train):
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
    
    if is_train:
        outputs = model(inputs_embeds=input_embedding, attention_mask=attention_mask)
    else:
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
        logits = get_logits(model, embed_layer, tokenizer, problem, label_ids, se, insert_idx, is_train=False)
        loss_val += cal_loss(logits, label_ids, target_tokens_list, last_tokens_list)
    loss_val /= len(eval_idxs)
    return loss_val


def get_cand_idx(dist_type, se_weight, embed_weight, beam):
    if dist_type == "l1":
        cand_idx = torch.topk(torch.cdist(se_weight.float(), embed_weight.float(), p=1), 
                              beam, dim=-1, largest=False).indices
    elif dist_type == "l2":
        cand_idx = torch.topk(torch.cdist(se_weight.float(), embed_weight.float(), p=2), 
                              beam, dim=-1, largest=False).indices
    elif dist_type == "cosine":
        cand_idx = torch.topk(F.normalize(se_weight, p=2, dim=-1) @ F.normalize(embed_weight, p=2, dim=-1).T, 
                              beam, dim=-1, largest=True).indices
    else:
        raise ValueError(f"Invalid distance type: {dist_type}")
    return cand_idx


def beam_search(model, embed_layer, tokenizer, data, eval_idxs, beam, cand_idx, 
                embed_weight, insert_idx, label_length, target_tokens_list, last_tokens_list):
    beam_loss_list = []
    best_idx = cand_idx[:, 0].reshape(-1)
    current_se_weight = embed_weight[best_idx]
    best_loss = get_eval_loss(model, embed_layer, tokenizer, data, eval_idxs, 
                              current_se_weight, insert_idx, label_length, 
                              target_tokens_list, last_tokens_list)
    beam_loss_list.append(best_loss)
    
    se_weight_copy = current_se_weight.clone()
    
    for pos in range(len(se_weight_copy)):
        for k in range(1, beam):
            se_weight_copy[pos] = embed_weight[cand_idx[pos, k]]
            loss = get_eval_loss(model, embed_layer, tokenizer, data, eval_idxs, 
                                se_weight_copy, insert_idx, label_length, 
                                target_tokens_list, last_tokens_list)
            beam_loss_list.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_idx[pos] = cand_idx[pos, k]
        se_weight_copy[pos] = embed_weight[best_idx[pos]]
    
    beam_loss_tensor = torch.stack(beam_loss_list).reshape(-1).sort().values
    final_se_weight = embed_weight[best_idx]
    
    return beam_loss_tensor, best_loss, best_idx, final_se_weight


def train_attack(
    model_path,
    data_path,
    config_path,
    output_dir,
    L=1,
    n_train_samples=20,
    n_eval_samples=10,
    seed=0,
    epochs=1000,
    lr=1e-3,
    label_length=2000,
    save_step=100,
    proj_step=0,
    dist_type="l2",
    beam=1,
    use_pca=False,
    n_components=50,
    gaussian_n=0,
    gaussian_std=0.02,
    penalty=0,
    penalty_topk=20,
):
    
    seed_everything(seed)
    
    model_name = model_path.split("/")[-1]
    print(f"Model: {model_name}")
    
    # Load data
    data = load_training_data(data_path)
    train_idxs = list(range(20))[:n_train_samples]
    eval_idxs = list(range(20, 30))[:n_eval_samples]
    print(f"Training samples: {len(train_idxs)}, Eval samples: {len(eval_idxs)}")
    
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
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Load target tokens
    target_tokens_list, last_tokens_list = load_model_tokens(config_path, model_name)
    target_tokens_list = [tokens.to(model.device) for tokens in target_tokens_list]
    last_tokens_list = [tokens.to(model.device) for tokens in last_tokens_list]
    
    # Get ASCII token embeddings
    ascii_token_ids = [token_id for token, token_id in tokenizer.get_vocab().items() 
                       if all(ord(c) < 128 for c in token)]
    ascii_token_ids = torch.tensor(ascii_token_ids, device=model.device)
    
    embed_layer = model.get_input_embeddings()
    embed_weight = embed_layer.weight[ascii_token_ids].data
    
    # Optional PCA
    pca = None
    embed_weight_pca = None
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        embed_weight_pca = pca.fit_transform(embed_weight.cpu().float().numpy())
        embed_weight_pca = torch.tensor(embed_weight_pca, device=model.device, dtype=torch.bfloat16)
        print(f"Using PCA with {n_components} components")
    
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_save = {
        'model_name': model_name,
        'L': L,
        'n_train_samples': n_train_samples,
        'n_eval_samples': n_eval_samples,
        'seed': seed,
        'epochs': epochs,
        'lr': lr,
        'proj_step': proj_step,
        'dist_type': dist_type,
        'use_pca': use_pca,
        'n_components': n_components if use_pca else 0,
        'gaussian_n': gaussian_n,
        'gaussian_std': gaussian_std,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_save, f, indent=2)
    
    # Initialize SE parameter
    se_param = nn.Parameter(torch.zeros(L, embed_weight.shape[-1], device=model.device, dtype=torch.bfloat16))
    se_param.data.normal_(mean=0.0, std=0.02)
    optimizer = Adam([se_param], lr=lr)
    
    insert_idx = get_insert_idx(tokenizer)
    
    # Training
    losses = []
    penalty_losses = []
    project_log = {'loss_before': [], 'loss_after': [], 'token_ids': []}
    
    print(f"Starting training for {epochs} iterations...")
    for t in tqdm(range(1, epochs + 1)):
        problem, label = get_qa_pair(data, random.choice(train_idxs))
        label_ids = tokenizer.encode(label, add_special_tokens=False, return_tensors='pt').to(model.device)
        label_ids = label_ids[0, :label_length]
        
        # Add Gaussian noise if specified
        if gaussian_n > 0:
            noise = torch.normal(mean=0.0, std=gaussian_std, 
                               size=(gaussian_n, *se_param.shape), 
                               device=se_param.device, dtype=se_param.dtype)
            logits = get_logits(model, embed_layer, tokenizer, problem, label_ids, 
                               se_param.unsqueeze(0) + noise, insert_idx, True).float()
        else:
            logits = get_logits(model, embed_layer, tokenizer, problem, label_ids, 
                               se_param, insert_idx, True).float()
        
        loss = cal_loss(logits, label_ids, target_tokens_list, last_tokens_list)
        total_loss = loss
        
        # Penalty term
        if penalty > 0:
            dist = torch.cdist(se_param.float(), embed_weight.float(), p=2)
            if penalty_topk > 0:
                dist, _ = torch.topk(dist, penalty_topk, dim=-1, largest=False)
            penalty_loss = penalty * dist.pow(2).mean()
            penalty_losses.append(penalty_loss.item())
            total_loss += penalty_loss
        
        if penalty > 0:
            print(f"t: {t}, train loss: {loss.item()}, penalty loss: {penalty_loss.item()}, total loss: {total_loss.item()}")
        else:
            print(f"t: {t}, train loss: {loss.item()}, total loss: {total_loss.item()}")
        
        losses.append(total_loss.item())
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Save checkpoint
        if t % save_step == 0:
            torch.save(se_param.data, output_dir / f'se_{t}.pt')
        
        # Projection step
        if proj_step > 0 and t % proj_step == 0 and t > 100:
            eval_loss_before = get_eval_loss(model, embed_layer, tokenizer, data, eval_idxs, 
                                            se_param.data, insert_idx, label_length, 
                                            target_tokens_list, last_tokens_list)
            
            if use_pca:
                se_weight_pca = pca.transform(se_param.data.cpu().float().numpy())
                se_weight_pca = torch.tensor(se_weight_pca, device=model.device, dtype=torch.bfloat16)
                cand_idx = get_cand_idx(dist_type, se_weight_pca, embed_weight_pca, beam)
            else:
                cand_idx = get_cand_idx(dist_type, se_param.data, embed_weight, beam)
            
            beam_loss_vals, best_loss, best_idx, se_weight_val = beam_search(
                model, embed_layer, tokenizer, data, eval_idxs, beam, cand_idx, 
                embed_weight, insert_idx, label_length, target_tokens_list, last_tokens_list
            )
            
            se_param.data = embed_weight[best_idx]
            
            project_log['loss_before'].append(eval_loss_before.item())
            project_log['loss_after'].append(best_loss.item())
            project_log['token_ids'].append(ascii_token_ids[best_idx].tolist())
    
    # Final projection
    if proj_step > 0:
        cand_idx = get_cand_idx(dist_type, se_param.data, embed_weight, beam)
        beam_loss_vals, best_loss, best_idx, se_weight_val = beam_search(
            model, embed_layer, tokenizer, data, eval_idxs, beam, cand_idx, 
            embed_weight, insert_idx, label_length, target_tokens_list, last_tokens_list
        )
        se_param.data = embed_weight[best_idx]
        losses.append(best_loss.item())
    
    # Save final results
    torch.save(se_param.data, output_dir / f'se_final.pt')
    
    with open(output_dir / 'losses.json', 'w') as f:
        json.dump(losses, f)
    
    if proj_step > 0:
        with open(output_dir / 'project_log.json', 'w') as f:
            json.dump(project_log, f)
    
    if penalty > 0:
        with open(output_dir / 'penalty_losses.json', 'w') as f:
            json.dump(penalty_losses, f)
    
    print(f"Training completed. Results saved to {output_dir}")
    
    return se_param.data


def main():
    parser = argparse.ArgumentParser(
        description="Train Deadlock Attack with projection-based optimization"
    )
    
    # Required arguments
    parser.add_argument("--model_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Path to the model")
    parser.add_argument("--data_path", type=str, 
                        default="data/generated_solutions.json",
                        help="Path to training data (generated by data_generation.py)")
    parser.add_argument("--config_path", type=str,
                        default="configs/model_tokens.yaml",
                        help="Path to model tokens config")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for checkpoints and logs")
    
    # Training arguments
    parser.add_argument("--L", type=int, default=1,
                        help="Length of special embedding")
    parser.add_argument("--n_train_samples", type=int, default=20,
                        help="Number of training samples")
    parser.add_argument("--n_eval_samples", type=int, default=10,
                        help="Number of evaluation samples")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Total training iterations")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--label_length", type=int, default=2000,
                        help="Maximum label length")
    parser.add_argument("--save_step", type=int, default=100,
                        help="Save checkpoint every N steps")
    
    # Projection arguments
    parser.add_argument("--proj_step", type=int, default=0,
                        help="Project to discrete tokens every N steps (0 = no projection)")
    parser.add_argument("--dist_type", type=str, default="l2",
                        choices=["l1", "l2", "cosine"],
                        help="Distance metric for projection")
    parser.add_argument("--beam", type=int, default=1,
                        help="Beam size for projection search")
    
    # PCA arguments
    parser.add_argument("--use_pca", action="store_true",
                        help="Use PCA for embedding projection")
    parser.add_argument("--n_components", type=int, default=50,
                        help="Number of PCA components")
    
    # Gaussian noise arguments
    parser.add_argument("--gaussian_n", type=int, default=0,
                        help="Number of Gaussian noise samples (0 = no noise)")
    parser.add_argument("--gaussian_std", type=float, default=0.02,
                        help="Standard deviation of Gaussian noise")
    
    # Penalty arguments
    parser.add_argument("--penalty", type=float, default=0,
                        help="Penalty coefficient for distance regularization")
    parser.add_argument("--penalty_topk", type=int, default=20,
                        help="Use top-k distances for penalty")
    
    args = parser.parse_args()
    
    train_attack(
        model_path=args.model_path,
        data_path=args.data_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        L=args.L,
        n_train_samples=args.n_train_samples,
        n_eval_samples=args.n_eval_samples,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        label_length=args.label_length,
        save_step=args.save_step,
        proj_step=args.proj_step,
        dist_type=args.dist_type,
        beam=args.beam,
        use_pca=args.use_pca,
        n_components=args.n_components,
        gaussian_n=args.gaussian_n,
        gaussian_std=args.gaussian_std,
        penalty=args.penalty,
        penalty_topk=args.penalty_topk,
    )


if __name__ == "__main__":
    main()