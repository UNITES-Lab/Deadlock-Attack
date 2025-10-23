# One Token Embedding Is Enough to Deadlock Your Large Reasoning Model

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/virtual/2025/poster/116766)

![Demo](assets/demo.gif)

## 📖 Overview

This repository contains the official implementation of our NeurIPS 2025 paper, which introduces a novel **deadlock attack** against Large Reasoning Models (LRMs). We demonstrate that a single, optimized adversarial token embedding can force state-of-the-art reasoning models into infinite reasoning loops. This attack causes the model to generate repetitive, non-progressive sequences of text that exhausts computational resources while remaining stealthy on normal inputs.

### Key Highlights

- 🎯 **Single Token Attack**: Only one adversarial embedding is needed to trigger the deadlock
- 🔥 **High Success Rate**: Achieves near perfect attack success rate on multiple LRMs
- 🚀 **Universal**: Works across different model architectures (Llama, Qwen, Phi)
- ⚡ **Efficient**: Low training cost with minimal computational resources
- 🧩 **Input-Agnostic**: The same adversarial token can deadlock the model regardless of the user's input query.
- 🎭 **Stealth**: Maintains model performance on clean inputs while exploiting reasoning vulnerabilities

## 🚀 Quick Start

### Installation

```bash
conda create -n attack python=3.11
conda activate attack
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install flask pyngrok datasets scikit-learn accelerate flash-attn fire setuptools==75.8.2 
pip install vllm==0.6.6.post1
```

## 📚 Usage

### 1. Generate Training Data

Generate question-solution pairs from existing datasets for training the adversarial embedding:

```bash
CUDA_VISIBLE_DEVICES=0 python data_generation.py \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --dataset_name "HuggingFaceH4/MATH-500" \
    --output_path "data/training_data.json" \
    --max_tokens 4000 \
    --num_samples 30 \
    --num_solutions 100 \
    --temperature 0.6 \
    --seed 42
```

### 2. Train Adversarial Embedding

Train the adversarial embedding that triggers the deadlock behavior:

```bash
CUDA_VISIBLE_DEVICES=0 python attack.py \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --data_path "data/training_data.json" \
    --output_dir "outputs/adversarial_embeddings" \
    --epochs 1000 \
    --L 1 \
    --n_train_samples 20 \
    --n_eval_samples 10 \
    --lr 1e-3 \
    --seed 0
```

### 3. Evaluate Attack Effectiveness

Evaluate the attack on benchmark datasets:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --data_name HuggingFaceH4/aime_2024 \
    --se_path "outputs/adversarial_embeddings/se.pt" \
    --output_dir "results" \
    --n_samples 50 \
    --max_tokens 20000 \
    --seed 42
```

### 4. Interactive Demo

Launch an interactive web interface to visualize the deadlock attack:

```bash
CUDA_VISIBLE_DEVICES=0 python app.py \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --se_path "outputs/adversarial_embeddings/se.pt" \
    --port 5000 \
    --use_ngrok
```

Then open your browser and navigate to `http://localhost:5000` to interact with the model.

### 5. Configuration

Target tokens and last tokens are pre-configured for each model in `configs/model_tokens.yaml`. To support a new model, you only need to customize the corresponding tokens in this configuration file.

## 📁 Project Structure

```
Deadlock-Attack/
├── app.py                      # Interactive web demo
├── attack.py                   # Adversarial embedding training
├── data_generation.py          # Training data generation
├── evaluate.py                 # End-to-end evaluation
├── LMC_eval.py                 # Linear Mode Connectivity (LMC) evaluation
├── requirements.txt            # Python dependencies
├── configs/
│   └── model_tokens.yaml      # Pre-configured target tokens for each model
├── HumanEval/                 # HumanEval benchmark utilities
│   ├── utils.py
│   ├── data/                  # Multi-language HumanEval datasets
│   └── human_eval/            # Evaluation scripts
└── assets/
    └── demo.gif               # Demo animation
```

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{zhang2025tokenembeddingdeadlocklarge,
      title={One Token Embedding Is Enough to Deadlock Your Large Reasoning Model}, 
      author={Mohan Zhang and Yihua Zhang and Jinghan Jia and Zhangyang Wang and Sijia Liu and Tianlong Chen},
      year={2025},
      eprint={2510.15965},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.15965}, 
}
```

## ⚠️ Responsible Use

This research is intended for academic purposes and to improve the security of Large Reasoning Models. Please use this code responsibly and ethically. We do not endorse malicious use of this attack against production systems.

## 🙏 Acknowledgments

We thank the open-source community for providing the foundation models and datasets used in this research:
- **DeepSeek-AI** for releasing the DeepSeek-R1 series, including **DeepSeek-R1-Distill-Qwen-7B** and **DeepSeek-R1-Distill-Llama-8B**.
- **Microsoft** for developing the Phi-4-Reasoning family, including **Phi-4-mini-reasoning**
- **NVIDIA** for the Llama-Nemotron series, including **Llama-3.1-Nemotron-Nano-8B-v1**
- **HuggingFace** for hosting and maintaining open access to the above models and benchmark datasets.