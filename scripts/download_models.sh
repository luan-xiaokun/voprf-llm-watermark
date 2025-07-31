#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --repo-type model Qwen/Qwen2.5-3B
huggingface-cli download --repo-type model Qwen/Qwen2.5-7B
