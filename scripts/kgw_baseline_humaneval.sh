#!/bin/bash
source .venv/bin/activate

# mkdir -p logs/kgw_baseline

# python src/humaneval_evaluation.py \
#     --delta 2.0 \
#     --gamma 0.25 \
#     --kgw_scheme lefthash \
#     2>&1 | tee logs/kgw_baseline/kgw_lefthash_humaneval_d2.0_g0.25.log

# python src/humaneval_evaluation.py \
#     --delta 2.0 \
#     --gamma 0.25 \
#     --kgw_scheme selfhash \
#     2>&1 | tee logs/kgw_baseline/kgw_selfhash_humaneval_d2.0_g0.25.log

# uv run src/gsm8k_evaluation.py --delta 2.0 --gamma 0.25 --window_size 7
uv run src/gsm8k_evaluation.py --delta 2.0 --gamma 0.25 --window_size 4

uv run src/gsm8k_evaluation.py --delta 2.0 --gamma 0.5 --window_size 4
uv run src/gsm8k_evaluation.py --delta 2.0 --gamma 0.5 --window_size 7

uv run src/gsm8k_evaluation.py --delta 2.0 --gamma 0.75 --window_size 4
uv run src/gsm8k_evaluation.py --delta 2.0 --gamma 0.75 --window_size 7

uv run src/gsm8k_evaluation.py --delta 3.0 --gamma 0.25 --window_size 4
uv run src/gsm8k_evaluation.py --delta 3.0 --gamma 0.25 --window_size 7

uv run src/gsm8k_evaluation.py --delta 3.0 --gamma 0.5 --window_size 4
uv run src/gsm8k_evaluation.py --delta 3.0 --gamma 0.5 --window_size 7

uv run src/gsm8k_evaluation.py --delta 3.0 --gamma 0.75 --window_size 4
uv run src/gsm8k_evaluation.py --delta 3.0 --gamma 0.75 --window_size 7