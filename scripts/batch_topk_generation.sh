#!/bin/bash
source .venv/bin/activate

mkdir -p logs/generation

window_sizes=(4 5 6 7 8)
deltas=(1.0)
gammas=(0.5)
top_ks=(10)

for top_k in "${top_ks[@]}"; do
    for gamma in "${gammas[@]}"; do
        for delta in "${deltas[@]}"; do
            for window_size in "${window_sizes[@]}"; do
                echo "Running with top_k=$top_k, gamma=$gamma, delta=$delta, window_size=$window_size"
                python src/generate_watermarked_text.py \
                    --num 500 \
                    --model_path Qwen/Qwen2.5-3B \
                    --output_dir output \
                    --window_size $window_size \
                    --delta $delta \
                    --gamma $gamma \
                    --max_tokens 210 \
                    --server_seed data/server_seed \
                    --batch_size 128 \
                    --do_sample \
                    --num_beams 1 \
                    --top_k $top_k \
                    --top_p 0.9 \
                    --temperature 0.7 \
                    --suppress_eos \
                    2>&1 | tee "logs/generation/Qwen2.5-3B_multinomial-top${top_k}_w${window_size}_d${delta}_g${gamma}.log"
            done
        done
    done
done
