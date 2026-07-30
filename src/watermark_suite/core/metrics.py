import math
from dataclasses import dataclass

import torch
import tqdm
from datasets import Dataset
from torch.nn import functional
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class PerplexityEvaluation:
    """Conditional perplexity results for a collection of continuations."""

    perplexity: float
    mean_negative_log_likelihood: float
    total_target_token_num: int
    sample_perplexities: list[float]
    sample_negative_log_likelihoods: list[float | None]
    sample_target_token_nums: list[int]


def _safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


def calculate_perplexities(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    prompt_column: str,
    target_column: str,
    batch_size: int = 2,
    max_length: int = 2048,
    device: str | torch.device | None = None,
) -> PerplexityEvaluation:
    """Calculate token-weighted and per-sample conditional perplexities.

    Prompt tokens are masked from the loss. When a sequence exceeds
    ``max_length``, the prompt is truncated from the left before any target
    tokens are removed, preserving as much of the evaluated continuation as
    possible.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(selected_device)
    model.eval()

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.bos_token_id
    if pad_token_id is None:
        raise ValueError("the tokenizer must define a pad, EOS, or BOS token")

    encoded_samples = []
    for sample in dataset:
        prompt_ids = tokenizer.encode(
            sample[prompt_column], add_special_tokens=False
        )
        target_ids = tokenizer.encode(
            sample[target_column], add_special_tokens=False
        )

        if len(target_ids) >= max_length:
            prompt_ids = []
            target_ids = target_ids[:max_length]
        else:
            prompt_budget = max_length - len(target_ids)
            prompt_ids = prompt_ids[-prompt_budget:]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if not input_ids:
            input_ids = [pad_token_id]
            labels = [-100]
        encoded_samples.append((input_ids, labels))

    sample_negative_log_likelihoods: list[float | None] = []
    sample_target_token_nums: list[int] = []
    total_negative_log_likelihood = 0.0
    total_target_tokens = 0

    with torch.no_grad():
        ranges = range(0, len(encoded_samples), batch_size)
        for start in tqdm.tqdm(ranges, desc="Calculating perplexity"):
            current = encoded_samples[start : start + batch_size]
            batch_max_length = max(len(input_ids) for input_ids, _ in current)
            input_batch = torch.full(
                (len(current), batch_max_length),
                pad_token_id,
                dtype=torch.long,
                device=selected_device,
            )
            attention_mask = torch.zeros(
                (len(current), batch_max_length),
                dtype=torch.long,
                device=selected_device,
            )
            label_batch = torch.full(
                (len(current), batch_max_length),
                -100,
                dtype=torch.long,
                device=selected_device,
            )

            for index, (input_ids, labels) in enumerate(current):
                length = len(input_ids)
                input_batch[index, :length] = torch.tensor(
                    input_ids, dtype=torch.long, device=selected_device
                )
                attention_mask[index, :length] = 1
                label_batch[index, :length] = torch.tensor(
                    labels, dtype=torch.long, device=selected_device
                )

            outputs = model(
                input_ids=input_batch,
                attention_mask=attention_mask,
                return_dict=True,
            )
            shift_logits = outputs.logits[:, :-1, :].float().contiguous()
            shift_labels = label_batch[:, 1:].contiguous()
            token_losses = functional.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.shape)
            target_mask = shift_labels != -100

            for index in range(len(current)):
                target_token_num = int(target_mask[index].sum().item())
                sample_target_token_nums.append(target_token_num)
                if target_token_num == 0:
                    sample_negative_log_likelihoods.append(None)
                    continue

                negative_log_likelihood = float(
                    token_losses[index][target_mask[index]].sum().item()
                )
                mean_nll = negative_log_likelihood / target_token_num
                sample_negative_log_likelihoods.append(mean_nll)
                total_negative_log_likelihood += negative_log_likelihood
                total_target_tokens += target_token_num

    sample_perplexities = [
        _safe_exp(value) if value is not None else float("inf")
        for value in sample_negative_log_likelihoods
    ]
    if total_target_tokens == 0:
        mean_negative_log_likelihood = float("inf")
        perplexity = float("inf")
    else:
        mean_negative_log_likelihood = (
            total_negative_log_likelihood / total_target_tokens
        )
        perplexity = _safe_exp(mean_negative_log_likelihood)

    return PerplexityEvaluation(
        perplexity=perplexity,
        mean_negative_log_likelihood=mean_negative_log_likelihood,
        total_target_token_num=total_target_tokens,
        sample_perplexities=sample_perplexities,
        sample_negative_log_likelihoods=(
            sample_negative_log_likelihoods
        ),
        sample_target_token_nums=sample_target_token_nums,
    )


def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    prompt_column: str,
    target_column: str,
    batch_size: int = 2,
    max_length: int = 2048,
) -> float:
    return calculate_perplexities(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        prompt_column=prompt_column,
        target_column=target_column,
        batch_size=batch_size,
        max_length=max_length,
    ).perplexity
