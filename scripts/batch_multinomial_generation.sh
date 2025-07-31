#!/bin/bash
source .venv/bin/activate

mkdir -p logs/generation

# window_sizes=(4 5 6 7 8 9)
# deltas=(0.5 1.0 2.0 3.0 4.0 5.0)
# gammas=(0.25 0.5 0.75)

window_sizes=(5 6 7 8)
deltas=(1.0 2.0 3.0 4.0)
gammas=(0.25 0.5 0.75)

for gamma in "${gammas[@]}"; do
    for delta in "${deltas[@]}"; do
        for window_size in "${window_sizes[@]}"; do
            echo "Running with gamma=$gamma, delta=$delta, window_size=$window_size"
            python src/generate_watermarked_text.py \
                --num 500 \
                --model_path Qwen/Qwen2.5-3B \
                --output_dir output \
                --window_size $window_size \
                --delta $delta \
                --gamma $gamma \
                --max_tokens 210 \
                --server_seed data/server_seed \
                --batch_size 64 \
                --do_sample \
                --num_beams 1 \
                --temperature 0.7 \
                2>&1 | tee "logs/generation/Qwen2.5-3B_multinomial-w${window_size}_d${delta}_g${gamma}.log"
        done
    done
done
