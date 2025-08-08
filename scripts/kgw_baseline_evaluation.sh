#!/bin/bash
source .venv/bin/activate

mkdir -p logs/kgw_baseline

# python src/baseline_evaluation.py \
#     --num 500 \
#     --model_path Qwen/Qwen2.5-3B \
#     --output_dir ./output/kgw_baseline \
#     --significance_level 0.01 \
#     --delta 2.0 \
#     --gamma 0.25 \
#     --seeding_scheme lefthash \
#     --max_tokens 210 \
#     --batch_size 4 \
#     --do_sample \
#     --num_beams 1 \
#     --temperature 0.7 \
#     --suppress_eos \
#     2>&1 | tee "logs/kgw_baseline/kgw_Qwen2.5-3B_multinomial_d2.0_g0.25.log"

python src/baseline_evaluation.py \
    --num 500 \
    --model_path Qwen/Qwen2.5-3B \
    --output_dir ./output/kgw_baseline \
    --significance_level 0.01 \
    --delta 2.0 \
    --gamma 0.25 \
    --seeding_scheme selfhash \
    --max_tokens 210 \
    --batch_size 4 \
    --do_sample \
    --num_beams 1 \
    --temperature 0.7 \
    --suppress_eos \
    2>&1 | tee "logs/kgw_baseline/kgw_selfhash_Qwen2.5-3B_multinomial_d2.0_g0.25.log"

python src/baseline_evaluation.py \
    --num 500 \
    --model_path Qwen/Qwen2.5-3B \
    --output_dir ./output/kgw_baseline \
    --significance_level 0.01 \
    --delta 2.0 \
    --gamma 0.25 \
    --seeding_scheme selfhash \
    --max_tokens 210 \
    --batch_size 4 \
    --do_sample \
    --top_k 10 \
    --top_p 0.9 \
    --num_beams 1 \
    --temperature 0.7 \
    --suppress_eos \
    2>&1 | tee "logs/kgw_baseline/kgw_selfhash_Qwen2.5-3B_multinomial-top10_d2.0_g0.25.log"

python src/baseline_evaluation.py \
    --num 500 \
    --model_path Qwen/Qwen2.5-3B \
    --output_dir ./output/kgw_baseline \
    --significance_level 0.01 \
    --delta 2.0 \
    --gamma 0.25 \
    --seeding_scheme selfhash \
    --max_tokens 210 \
    --batch_size 4 \
    --do_sample \
    --top_k 50 \
    --top_p 0.9 \
    --num_beams 1 \
    --temperature 0.7 \
    --suppress_eos \
    2>&1 | tee "logs/kgw_baseline/kgw_selfhash_Qwen2.5-3B_multinomial-top50_d2.0_g0.25.log"

python src/baseline_evaluation.py \
    --num 500 \
    --model_path Qwen/Qwen2.5-3B \
    --output_dir ./output/kgw_baseline \
    --significance_level 0.01 \
    --delta 2.0 \
    --gamma 0.25 \
    --seeding_scheme selfhash \
    --max_tokens 210 \
    --batch_size 4 \
    --do_sample \
    --top_k 100 \
    --top_p 0.9 \
    --num_beams 1 \
    --temperature 0.7 \
    --suppress_eos \
    2>&1 | tee "logs/kgw_baseline/kgw_selfhash_Qwen2.5-3B_multinomial-top100_d2.0_g0.25.log"
