import math

import torch
import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator, PreTrainedModel, PreTrainedTokenizerBase


def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    prompt_column: str,
    target_column: str,
    batch_size: int = 2,
    max_length: int = 2048,
) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def preprocess(examples):
        prompts = examples[prompt_column]
        targets = examples[target_column]
        full_texts = [p + t for p, t in zip(prompts, targets)]

        full_tokenized = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        prompt_tokenized = tokenizer(prompts, add_special_tokens=False)

        labels = []
        for i in range(len(full_tokenized["input_ids"])):
            prompt_len = len(prompt_tokenized["input_ids"][i])

            label = list(full_tokenized["input_ids"][i])

            label[:prompt_len] = [-100] * prompt_len
            labels.append(label)
        full_tokenized["labels"] = torch.tensor(labels)

        return full_tokenized

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=DefaultDataCollator(),
        shuffle=False,
    )

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Calculating perplexity"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            num_target_tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * num_target_tokens
            total_tokens += num_target_tokens

    if total_tokens == 0:
        return float("inf")

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)

    return perplexity
