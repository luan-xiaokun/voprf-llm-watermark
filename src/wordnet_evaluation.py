import utils
from detection import local_detect_batch_text
from transformers import AutoTokenizer
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score


def main():
    gamma = 0.25
    watermarked_texts = defaultdict(list)
    attacked_texts = defaultdict(list)
    unwatermarked_texts = set()
    watermarked_detection_results = {}
    attacked_detection_results = {}

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    with open("data/server_seed") as f:
        seed = bytes.fromhex(f.read().strip())
    window_sizes = set()
    expected_token_nums = set()
    for sample in utils.read_jsonlines("data/wn_attacked_result_8k.jsonl"):
        window_size = sample["window_size"]
        generation = sample["generation"]
        regeneration = sample["regeneration"]
        expected_token_num = sample["expected_token_num"]
        watermarked_texts[(expected_token_num, window_size)].append(generation)
        attacked_texts[(expected_token_num, window_size)].append(regeneration)
        window_sizes.add(window_size)
        expected_token_nums.add(expected_token_num)
        unwatermarked_texts.add(sample["answer"])

    for token_num in expected_token_nums:
        w_results = []
        a_results = []
        for window_size in window_sizes:
            watermarked_texts_list = watermarked_texts[(token_num, window_size)]
            attacked_texts_list = attacked_texts[(token_num, window_size)]

            print(f"Processing token_num={token_num}, window_size={window_size}")

            watermarked_detection_result = local_detect_batch_text(
                watermarked_texts_list, tokenizer, window_size, gamma, seed
            )
            attacked_detection_result = local_detect_batch_text(
                attacked_texts_list, tokenizer, window_size, gamma, seed
            )

            valid_indices = [
                i
                for i, r in enumerate(watermarked_detection_result)
                if r.p_value < 1e-5
            ]

            w_results.extend([watermarked_detection_result[i] for i in valid_indices])
            a_results.extend([attacked_detection_result[i] for i in valid_indices])
        watermarked_detection_results[token_num] = w_results
        attacked_detection_results[token_num] = a_results

    unwatermarked_detection_results = local_detect_batch_text(
        list(unwatermarked_texts), tokenizer, 4, gamma, seed
    )
    unwatermarked_p_values = [r.p_value for r in unwatermarked_detection_results]

    sorted_token_nums = list(sorted(expected_token_nums))

    watermarked_p_values_by_token_num = [
        [r.p_value for r in watermarked_detection_results[token_num]]
        for token_num in sorted_token_nums
    ]
    attacked_p_values_by_token_num = [
        [r.p_value for r in attacked_detection_results[token_num]]
        for token_num in sorted_token_nums
    ]
    watermarked_p_values_by_token_num.append(unwatermarked_p_values)

    plt.boxplot(watermarked_p_values_by_token_num, tick_labels=sorted_token_nums + ["Negative"])
    plt.savefig("figures/watermarked_detection_results_before_wn.png", dpi=300)

    plt.boxplot(attacked_p_values_by_token_num, tick_labels=sorted_token_nums)
    plt.savefig("figures/attacked_detection_results_after_wn.png", dpi=300)

    for w_p_vals, a_p_vals in zip(
        watermarked_p_values_by_token_num, attacked_p_values_by_token_num
    ):
        auc_w = roc_auc_score(
            [1] * len(w_p_vals) + [2] * len(unwatermarked_p_values),
            w_p_vals + unwatermarked_p_values,
        )
        auc_a = roc_auc_score(
            [1] * len(a_p_vals) + [2] * len(unwatermarked_p_values),
            a_p_vals + unwatermarked_p_values,
        )
        print(f"AUC (watermarked): {auc_w:.4f}, AUC (attacked): {auc_a:.4f}")


if __name__ == "__main__":
    main()
