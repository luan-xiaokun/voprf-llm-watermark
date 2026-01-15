import numpy as np
import torch
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from watermark_suite.schemes.upv import UPVDetector, UPVAdapter

MODEL_PATH = "Qwen/Qwen2.5-0.5B"
STRIDE = 255


def int_to_bin_list(n, length=8):
    bin_str = format(n, "b").zfill(length)
    return [int(b) for b in bin_str]


def main():
    total_num = 1_000_000
    window_size = 4
    gamma = 0.5
    bits_num = 18

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    vocab_size = tokenizer.vocab_size
    detector = UPVDetector(
        tokenizer, "model", window_size=window_size, bits_num=bits_num, gamma=gamma
    )
    adapter = UPVAdapter(
        model,
        tokenizer,
        "model",
        window_size,
        delta=2.0,
        bit_number=bits_num,
        layers=5,
        beam_size=0,
    )

    dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)

    print(f"Using model: {MODEL_PATH}")
    print(f"Gamma: {gamma}")
    print(f"Vocab size: {vocab_size}")

    private_confidence_scores = []
    pbar = tqdm.tqdm(total=total_num, desc="Processing samples")
    for sample in dataset:
        text = sample["text"]
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        batch_size = 32
        idxs = list(range(0, len(token_ids) - STRIDE, STRIDE))
        for j in range(0, len(idxs), batch_size):
            batch_idxs = idxs[j : j + batch_size]
            batch_chunks = [token_ids[i : i + STRIDE] for i in batch_idxs]
            batch_inputs_bin = [
                [int_to_bin_list(n, bits_num) for n in chunk] for chunk in batch_chunks
            ]
            inputs_tensor = torch.tensor(batch_inputs_bin, dtype=torch.float32).to(
                detector.device
            )
            with torch.no_grad():
                outputs = detector.provider_detector_model(inputs_tensor)
                private_confidence_scores.extend(
                    outputs.reshape(-1).cpu().numpy().tolist()
                )
            pbar.update(len(batch_idxs))
        if len(private_confidence_scores) >= total_num:
            break
    pbar.close()

    array = np.array(private_confidence_scores)

    false_positive_num = np.sum(array > 0.5)
    fpr = false_positive_num / len(array)
    print(f"Private Detector False Positive Rate (FPR): {fpr:.6f}")

    validation_data_dir = Path("data")
    validation_data_dir.mkdir(parents=True, exist_ok=True)
    with open(validation_data_dir / "validation_confidence_scores_new.npy", "wb") as f:
        np.save(f, array)

    provider_z_scores = []
    false_positive_num = 0
    pbar = tqdm.tqdm(total=total_num, desc="Processing samples")
    for sample in dataset:
        text = sample["text"]
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        batch_size = 32
        idxs = list(range(0, len(token_ids) - STRIDE, STRIDE))
        for j in range(0, len(idxs), batch_size):
            batch_idxs = idxs[j : j + batch_size]
            batch_chunks = [token_ids[i : i + STRIDE] for i in batch_idxs]
            z_scores = adapter.batch_green_token_z_scores(batch_chunks)
            false_positive_num += torch.sum(z_scores > 4.0).item()
            provider_z_scores.extend(z_scores.cpu().numpy().tolist())
            pbar.update(len(batch_idxs))
            cur_fpr = false_positive_num / len(provider_z_scores)
            pbar.set_postfix({"FPR": f"{cur_fpr * 100:.4f}%"})
        if len(provider_z_scores) >= total_num:
            break

    array = np.array(provider_z_scores)

    # use 4.0 as threshold to calculate FPR
    false_positive_num = np.sum(array > 4.0)
    fpr = false_positive_num / len(array)
    print(f"Provider False Positive Rate (FPR): {fpr:.6f}")

    validation_data_dir = Path("data")
    validation_data_dir.mkdir(parents=True, exist_ok=True)
    with open(validation_data_dir / "validation_provider_z_scores_new.npy", "wb") as f:
        np.save(f, array)

    # validation_data_dir = Path("data")
    # # load private_confidence_scores
    # private_confidence_scores = np.load(
    #     validation_data_dir / "validation_confidence_scores.npy"
    # )
    # provider_z_scores = np.load(
    #     validation_data_dir / "validation_provider_z_scores.npy"
    # )

    # assert len(private_confidence_scores) == len(provider_z_scores)
    # private_fpr = np.sum(private_confidence_scores > 0.5) / len(
    #     private_confidence_scores
    # )
    # provider_fpr = np.sum(provider_z_scores > 4.0) / len(provider_z_scores)
    # print(f"Private Detector FPR: {private_fpr * 100:.4f}%")
    # print(f"Provider Detector FPR: {provider_fpr * 100:.4f}%")

    # # compare the two arrays and check for inconsistencies
    # incons = 0
    # for pc, pz in zip(private_confidence_scores, provider_z_scores):
    #     private_detected = pc > 0.5
    #     provider_detected = pz > 4.0
    #     if private_detected != provider_detected:
    #         incons += 1

    # print(f"Inconsistencies between detectors: {incons}")
    # print(f"Inconsistency Rate: {incons / len(private_confidence_scores) * 100:.4f}%")


if __name__ == "__main__":
    main()


# Private Detector FPR: 0.7920%
# Provider Detector FPR: 0.1748%
# Inconsistencies between detectors: 6794
# Inconsistency Rate: 0.6794%

# # For negative samples (no watermark)
# |                   | Private Positive | Private Negative |
# |-------------------|------------------|------------------|
# | Provider Positive |       1437       |       311        |
# | Provider Negative |       6483       |      991769      |

# This means that:
# 1. 99.1769% of negative samples are correctly identified as negative by both detectors.
# 2. 0.6783% of negative samples are identified as positive by private detectors, but negative by provider detector.
# 3. 0.1437% of negative samples are incorrectly identified as positive by both detectors.
# 4. 0.0311% of negative samples are identified as positive by provider detector, but negative by private detectors.
