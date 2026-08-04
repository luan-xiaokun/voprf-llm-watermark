from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import DynamicCache

from watermark_suite.schemes.pdw import generate
from watermark_suite.schemes.pdw import watermarking


class _CacheMutatingModel:
    def __init__(self):
        self.past_lengths = []

    def __call__(self, inputs, *, past_key_values, attention_mask):
        del attention_mask
        self.past_lengths.append(past_key_values.get_seq_length())
        state = torch.zeros((1, 1, inputs.shape[1], 1))
        past_key_values.update(state, state, 0)
        logits = torch.full((1, inputs.shape[1], 4), -1000.0)
        logits[:, -1, len(self.past_lengths)] = 1000.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=past_key_values,
        )


class _CharacterTokenizer:
    def decode(self, token_ids):
        values = torch.as_tensor(token_ids).reshape(-1).tolist()
        return "".join("ABCD"[int(value)] for value in values)


class _NoOpMonitor:
    def record_step_start(self):
        pass

    def record_step_end(self, *, is_prefill):
        del is_prefill


def test_pdw_downstream_generation_uses_stochastic_candidates(monkeypatch):
    def generate_text(
        prompt,
        model,
        tokenizer,
        sample_type,
        *args,
        **kwargs,
    ):
        del model, tokenizer, args, kwargs
        if sample_type == "argmax":
            raise ValueError(
                "tried to plant another error but already at "
                "max_planted_errors: 2"
            )
        return prompt.upper(), torch.tensor([]), b"", (), 0, 0

    monkeypatch.setattr(watermarking, "generate_text_asymmetric", generate_text)
    adapter = watermarking.PDWAdapter(model=object(), tokenizer=object())

    texts = adapter(
        prompts=["watermark me"],
        max_new_tokens=32,
        do_sample=False,
        seed=17,
    )

    assert texts == ["WATERMARK ME"]


def test_pdw_retries_with_distinct_reproducible_seeds(monkeypatch):
    observed_seeds = []

    def generate_text(*args, **kwargs):
        del args, kwargs
        observed_seeds.append(
            (
                torch.initial_seed(),
                int(np.random.get_state()[1][0]),
                random.getstate()[1][0],
            )
        )
        if len(observed_seeds) % 3 != 0:
            raise ValueError(
                "tried to plant another error but already at "
                "max_planted_errors: 2"
            )
        return "forged", torch.tensor([]), b"", (), 0, 0

    monkeypatch.setattr(watermarking, "generate_text_asymmetric", generate_text)
    adapter = watermarking.PDWAdapter(
        model=object(),
        tokenizer=object(),
        max_generation_attempts=3,
    )

    first = adapter(
        prompts=["watermark me"],
        max_new_tokens=32,
        seed=17,
    )
    first_seeds = list(observed_seeds)
    observed_seeds.clear()
    second = adapter(
        prompts=["watermark me"],
        max_new_tokens=32,
        seed=17,
    )

    assert first == second == ["forged"]
    assert len(set(first_seeds)) == 3
    assert observed_seeds == first_seeds


def test_pdw_reports_retry_exhaustion(monkeypatch):
    calls = 0

    def generate_text(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise ValueError(
            "tried to plant another error but already at "
            "max_planted_errors: 2"
        )

    monkeypatch.setattr(watermarking, "generate_text_asymmetric", generate_text)
    adapter = watermarking.PDWAdapter(
        model=object(),
        tokenizer=object(),
        max_generation_attempts=3,
    )

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        adapter(prompts=["watermark me"], max_new_tokens=32, seed=17)

    assert calls == 3


def test_pdw_does_not_retry_non_stochastic_errors(monkeypatch):
    calls = 0

    def generate_text(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise ValueError("invalid PDW key material")

    monkeypatch.setattr(watermarking, "generate_text_asymmetric", generate_text)
    adapter = watermarking.PDWAdapter(
        model=object(),
        tokenizer=object(),
        max_generation_attempts=3,
    )

    with pytest.raises(ValueError, match="invalid PDW key material"):
        adapter(prompts=["watermark me"], max_new_tokens=32, seed=17)

    assert calls == 1


def test_pdw_retries_signature_candidates_from_the_same_cache(monkeypatch):
    model = _CacheMutatingModel()
    cache = DynamicCache()
    state = torch.zeros((1, 1, 1, 1))
    cache.update(state, state, 0)
    monkeypatch.setattr(
        generate.crypto,
        "sign_and_encode_openssl",
        lambda *args, **kwargs: "0",
    )
    monkeypatch.setattr(
        generate.crypto,
        "get_signature_codeword_length",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        generate.crypto,
        "unkeyed_hash_to_bits",
        lambda value, bit_size: "0" if value.endswith(b"D") else "1",
    )
    monkeypatch.setattr(generate, "tqdm", lambda values: values)

    text, *_ = generate.generate_message_signature_pair(
        message_length=1,
        signature_segment_length=1,
        bit_size=1,
        max_planted_errors=1,
        sk=[],
        params=(),
        model=model,
        tokenizer=_CharacterTokenizer(),
        vocab_size=4,
        sample_type="multinomial",
        inputs=torch.tensor([[0, 0]]),
        past=cache,
        attn=torch.ones((1, 2), dtype=torch.long),
        counter=0,
        monitor=_NoOpMonitor(),
        embedded_first_message_signature_pair=False,
    )

    assert text == "BD"
    assert model.past_lengths == [1, 2, 2]
