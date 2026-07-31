from __future__ import annotations

from watermark_suite.schemes.rdf.watermarking import RDFAdapter, shift_generate
from watermark_suite.schemes.vow.rejection_sampling import (
    recover_sampling_mixin,
    replace_with_rejection_sampling_mixin,
)


class FakeModel:
    def _sample(
        self,
        input_ids,
        logits_processor=None,
        **model_kwargs,
    ):
        return input_ids, logits_processor, model_kwargs


class FakeTokenizer:
    @staticmethod
    def get_vocab():
        return {"left": 0, "right": 1}


def test_recovery_restores_original_bound_sampling_method():
    model = FakeModel()
    original_sample = model._sample

    replaced = replace_with_rejection_sampling_mixin(
        model,
        window_size=4,
        delta=2.5,
        gamma=0.5,
        voprf_server=object(),
    )
    recover_sampling_mixin(model, replaced)

    result = model._sample(
        "input-ids",
        logits_processor="processor",
    )

    assert result == ("input-ids", "processor", {})
    assert model._sample == original_sample


def test_rdf_recovery_restores_generate_without_overwriting_sample():
    model = FakeModel()
    model.generate = lambda **kwargs: kwargs
    original_generate = model.generate
    original_sample = model._sample
    adapter = RDFAdapter(
        model,
        FakeTokenizer(),
        watermark_sequence_length=8,
        seed=42,
    )

    adapter._pre_generation(
        do_sample=False,
        num_beams=1,
        top_p=None,
        top_k=None,
        no_watermark=False,
    )

    assert model.generate.__func__ is shift_generate
    adapter._post_generation()
    assert model.generate is original_generate
    assert model._sample == original_sample
    assert not hasattr(model, "watermark_n")
