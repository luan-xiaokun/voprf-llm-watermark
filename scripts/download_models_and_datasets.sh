#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --repo-type model Qwen/Qwen2.5-3B
huggingface-cli download --repo-type model Qwen/Qwen2.5-3B-Instruct
huggingface-cli download --repo-type model Qwen/Qwen2.5-7B
huggingface-cli download --repo-type model Qwen/Qwen2.5-7B-Instruct
huggingface-cli download --repo-type model Qwen/Qwen2.5-14B
huggingface-cli download --repo-type model distilbert-base-uncased
huggingface-cli download --repo-type model sentence-transformers/all-mpnet-base-v2
huggingface-cli download --repo-type dataset openai/gsm8k
huggingface-cli download --repo-type dataset sentence-transformers/eli5 --local-dir data/eli5
