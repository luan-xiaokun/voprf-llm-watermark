from __future__ import annotations

import torch

from watermark_suite.schemes.upv.detection import (
    TransformerClassifier,
    UPVDetector,
)


class _Tokenizer:
    def __call__(self, text, *, return_tensors, add_special_tokens):
        del text
        assert return_tensors == "pt"
        assert add_special_tokens is True
        return {"input_ids": torch.arange(1, 13).unsqueeze(0)}


def test_upv_detection_uses_an_explicit_classifier_decision_not_a_p_value():
    model = TransformerClassifier(
        bit_number=18,
        b_layers=5,
        input_dim=64,
        hidden_dim=128,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    model.fc.bias.data.fill_(1.0)
    model.eval()

    detector = object.__new__(UPVDetector)
    detector.tokenizer = _Tokenizer()
    detector.window_size = 4
    detector.bits_num = 18
    detector.gamma = 0.5
    detector.provider_detector_model = model
    detector.device = "cpu"

    result = detector.detect("watermarked", step_size=5)

    assert result.p_value is None
    assert result.score_type == "classifier_confidence"
    assert result.decision_threshold == 0.5
    assert result.predicted is True
    assert result.milestones == [5, 10]
    assert result.step_predictions == [True, True]
    assert result.step_p_values is None
