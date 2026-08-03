from __future__ import annotations

from types import SimpleNamespace

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
