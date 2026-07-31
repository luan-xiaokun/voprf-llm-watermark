from __future__ import annotations

import json

import pytest

from watermark_suite.experiments.errors import ResolutionError
from watermark_suite.experiments.dataset_prompt import DATASET_PROMPTS
from watermark_suite.experiments.runtime import (
    LocalModelRuntime,
    verify_model_materials,
)
from watermark_suite.experiments.scheme_registry import WATERMARK_SCHEMES
from watermark_suite.experiments.stages.common import resolve_model


def test_selected_sample_id_cannot_be_overridden_by_source_record(tmp_path):
    dataset_path = tmp_path / "c4.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "original_index": 7,
                "sample_id": "source-controlled",
                "prompt_text": "prompt",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    population = DATASET_PROMPTS.resolve(
        {
            "kind": "c4",
            "format": "jsonl",
            "path": str(dataset_path),
        },
        catalog={},
        repository=tmp_path,
        sample_num=1,
    )

    samples = DATASET_PROMPTS.materialize(population).samples

    assert samples[0].sample_id == "c4:7"


def test_local_model_material_is_reverified_before_runtime_load(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"original")
    model = resolve_model(
        {"checkpoint": str(checkpoint)},
        models={},
        repository=tmp_path,
    )
    weights.write_bytes(b"changed")

    with pytest.raises(ResolutionError, match="changed after Plan resolution"):
        verify_model_materials(model)


def test_runtime_reverifies_material_after_release(monkeypatch):
    import watermark_suite.experiments.runtime as runtime_module

    model = {
        "location": "/checkpoint",
        "verification": {"kind": "directory-snapshot", "digest": "model"},
        "tokenizer_location": "/checkpoint",
        "tokenizer_verification": {
            "kind": "directory-snapshot",
            "digest": "model",
        },
    }
    verified: list[dict] = []
    monkeypatch.setattr(
        runtime_module,
        "verify_model_materials",
        lambda value: verified.append(value),
    )
    runtime = LocalModelRuntime()

    runtime._verify(model)
    runtime._verify(model)
    runtime.release()
    runtime._verify(model)

    assert verified == [model, model]


def test_local_model_cannot_claim_an_unverified_revision(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")

    with pytest.raises(ResolutionError, match="declared revision"):
        resolve_model(
            {
                "checkpoint": str(checkpoint),
                "revision": "claimed-commit",
            },
            models={},
            repository=tmp_path,
        )


@pytest.mark.parametrize("method", ["pdw", "upv"])
def test_external_watermark_material_is_reverified(method, tmp_path):
    if method == "pdw":
        settings = {
            "method": "pdw",
            "enabled": True,
            "sk_path": str(tmp_path / "sk"),
            "pk_path": str(tmp_path / "pk"),
            "params_path": str(tmp_path / "params"),
            "signature_segment_length": 2,
            "bit_size": 2,
            "message_length": 2,
            "max_planted_errors": 0,
            "seed": 1,
        }
        for name in ("sk", "pk", "params"):
            (tmp_path / name).write_bytes(name.encode())
        changed = tmp_path / "pk"
    else:
        detector_dir = tmp_path / "upv"
        detector_dir.mkdir()
        for name in ("combine_model.pt", "private_detector.pt"):
            (detector_dir / name).write_bytes(name.encode())
        settings = {
            "method": "upv",
            "enabled": True,
            "detector_dir": str(detector_dir),
            "window_size": 4,
            "delta": 2.0,
            "gamma": 0.5,
            "bit_number": 16,
            "layers": 5,
            "beam_size": 1,
        }
        changed = detector_dir / "private_detector.pt"
    watermark = WATERMARK_SCHEMES.resolve(settings, tmp_path)
    changed.write_bytes(b"changed")

    with pytest.raises(ResolutionError, match="changed after Plan resolution"):
        WATERMARK_SCHEMES.verify(watermark)
