import time
import types
import warnings

import torch
from transformers.generation import GenerationConfig, LogitsProcessorList
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..adapter import WatermarkAdapter
from .mersenne import mersenne_rng


class RDFAdapter(WatermarkAdapter):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        watermark_sequence_length: int = 256,
        seed: int = 42,
        watermark_sequence_device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.watermark_sequence_length = watermark_sequence_length
        self.seed = seed
        self.watermark_sequence_device = watermark_sequence_device
        self.original_generate = None

    def _pre_generation(
        self,
        do_sample: bool,
        num_beams: int,
        top_p: float | None,
        top_k: float | None,
        no_watermark: bool,
    ) -> None:
        if no_watermark:
            return
        if num_beams is not None and num_beams > 1:
            warnings.warn(
                "RDF watermarking is enabled, beam search (num_beams > 1) will be ignored."
            )
        if do_sample:
            warnings.warn(
                "RDF watermarking is enabled, sampling strategies (do_sample, top_p, top_k, temperature) will be ignored."
            )

        self.original_generate = apply_shift_watermark(
            self.model,
            self.tokenizer,
            self.watermark_sequence_length,
            self.seed,
            self.watermark_sequence_device,
        )

    def _post_generation(self) -> None:
        if self.original_generate is not None:
            self.model._sample = types.MethodType(self.original_generate, self.model)
            self.original_generate = None

            del self.model.watermark_n
            del self.model.watermark_key
            del self.model.vocab_size
            del self.model.watermark_sequence_device


def shift_generate(
    self: PreTrainedModel,
    input_ids: torch.Tensor,
    generation_config: GenerationConfig | None,
    logits_processor: LogitsProcessorList | None,
    **kwargs,
):
    start = time.perf_counter()
    if generation_config is not None:
        max_new_tokens = generation_config.max_new_tokens
        eos_token_id = generation_config.eos_token_id
        pad_token_id = generation_config.pad_token_id
    else:
        max_new_tokens = kwargs.get("max_new_tokens")
        eos_token_id = kwargs.get("eos_token_id")
        pad_token_id = kwargs.get("pad_token_id")

    eos_token_id = eos_token_id or self.config.eos_token_id
    pad_token_id = pad_token_id or (
        self.config.pad_token_id
        if self.config.pad_token_id is not None
        else eos_token_id
    )

    n = self.watermark_n
    key = self.watermark_key
    vocab_size = self.vocab_size

    batch_size = input_ids.shape[0]
    inputs = input_ids.to(self.device)
    watermark_device = self.watermark_sequence_device

    rng = mersenne_rng(key)
    xi = torch.tensor(
        [rng.rand() for _ in range(n * vocab_size)], device=watermark_device
    ).view(n, vocab_size)
    shifts = torch.randint(
        n, (batch_size,), device=watermark_device
    )  # Shape: (batch_size,)

    unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=self.device)

    past_key_values = None
    if "attention_mask" in kwargs:
        attention_mask = kwargs["attention_mask"].to(self.device)
    else:
        attention_mask = torch.ones_like(inputs)

    end = time.perf_counter()
    print(f"Time taken before entering loop: {end - start:.4f} seconds")

    step = 0
    while True:
        if step >= max_new_tokens or not unfinished_sequences.any():
            break

        model_inputs = inputs[:, -1:] if past_key_values else inputs

        with torch.no_grad():
            outputs = self(
                model_inputs,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
            )

        # this is only for timing purpose
        if logits_processor is not None:
            logits_processor(inputs, outputs.logits)

        probs = torch.nn.functional.softmax(
            outputs.logits[:, -1, :vocab_size], dim=-1
        ).to(device=watermark_device)

        current_indices = (shifts + step) % n
        u_vectors = xi[current_indices, :]  # Shape: (batch_size, vocab_size)

        next_tokens = exp_sampling(probs, u_vectors).to(
            self.device
        )  # Shape: (batch_size, 1)

        newly_finished = next_tokens.view(-1) == eos_token_id
        unfinished_sequences.masked_fill_(newly_finished, 0)

        next_tokens = next_tokens * unfinished_sequences.unsqueeze(
            -1
        ) + pad_token_id * (~unfinished_sequences).unsqueeze(-1)

        inputs = torch.cat([inputs, next_tokens], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1
        )
        past_key_values = outputs.past_key_values

        step += 1

    return inputs.detach()


def exp_sampling(probs, u):
    return torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)


def apply_shift_watermark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    key: int,
    watermark_sequence_device: str,
):
    model.watermark_n = n
    model.watermark_key = key
    model.vocab_size = len(tokenizer.get_vocab())
    model.watermark_sequence_device = watermark_sequence_device

    original_generate = model.generate

    model.generate = types.MethodType(shift_generate, model)

    return original_generate
