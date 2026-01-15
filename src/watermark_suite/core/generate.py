from collections.abc import Generator
from typing import Callable

import torch
from datasets import Dataset
from tqdm import tqdm

from ..schemes.adapter import WatermarkAdapter


def generate_texts(
    adapter: WatermarkAdapter,
    dataset: Dataset,
    batch_size: int,
    prompt_column: str,
    max_new_tokens: int | None = None,
    do_sample: bool = False,
    top_p: float | None = None,
    top_k: int | None = None,
    temperature: float | None = None,
    suppress_tokens: int | list[int] | None = None,
    stop_strings: str | list[str] | None = None,
    prompt_formatter: Callable[[str], str] | None = None,
    no_watermark: bool = False,
) -> Generator[dict, None, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter.to(device)
    adapter.eval()

    # print("Warming up...")
    # warmup_prompt = ["Just a test to warm up the GPU"]
    # _ = adapter(
    #     prompts=warmup_prompt,
    #     max_new_tokens=8,
    #     pad_token_id=adapter.tokenizer.eos_token_id,
    #     no_watermark=no_watermark,
    # )
    # torch.cuda.synchronize()
    # print("Warm-up finished")

    if prompt_formatter is None:
        prompt_formatter = lambda x: x

    # iterate over the dataset and generate watermarked text
    generation_total_time = 0.0
    generated_total_tokens = 0
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    total_batches = (len(dataset) + batch_size - 1) // batch_size
    for batch in (
        pbar := tqdm(dataset.iter(batch_size=batch_size), total=total_batches)
    ):
        prompts = [prompt_formatter(prompt) for prompt in batch[prompt_column]]
        with torch.no_grad():
            starter.record()
            texts = adapter(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                suppress_tokens=suppress_tokens,
                pad_token_id=adapter.tokenizer.eos_token_id,
                no_watermark=no_watermark,
            )
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000.0
            generation_total_time += curr_time

        batch["generated_text"] = texts

        generated_total_tokens += sum(
            len(adapter.tokenizer.encode(text, add_special_tokens=False))
            for text in texts
        )
        avg_tps = generated_total_tokens / generation_total_time
        pbar.set_description(f"Avg tokens per second: {avg_tps:.2f} t/s")

        yield batch

    final_avg_tps = generated_total_tokens / generation_total_time
    print(
        f"Generated {len(dataset)} completions in {generation_total_time:.2f} seconds"
    )
    print(f"Average tokens per second: {final_avg_tps:.2f} t/s")

    return final_avg_tps
