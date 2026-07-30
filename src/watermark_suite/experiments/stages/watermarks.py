from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from watermark_suite.schemes import (
    KGWAdapter,
    KGWDetector,
    PDWAdapter,
    PDWDetector,
    RDFAdapter,
    RDFDetector,
    VOWAdapter,
    VOWDetector,
    WatermarkAdapter,
)
from watermark_suite.schemes.upv import UPVAdapter, UPVDetector
from watermark_suite.schemes.vow.key import get_server_seed

from ..errors import PlanValidationError, ResolutionError
from ..identity import sha256_file
from ..models import JsonObject
from .common import resolve_path


def _file(
    value: str, repository: Path, *, label: str
) -> tuple[str, str]:
    path = resolve_path(value, repository)
    if not path.is_file():
        raise ResolutionError(f"{label} does not exist: {path}")
    return str(path), sha256_file(path)


def resolve_watermark(value: Any, repository: Path) -> JsonObject:
    if not isinstance(value, dict):
        raise PlanValidationError("watermark must be a mapping")
    method = value.get("method")
    allowed: dict[str, set[str]] = {
        "none": {"method", "enabled"},
        "vow": {
            "method",
            "enabled",
            "window_size",
            "delta",
            "gamma",
            "server_seed_path",
            "naive_baseline",
        },
        "lefthash": {"method", "enabled", "delta", "gamma"},
        "selfhash": {"method", "enabled", "delta", "gamma"},
        "rdf": {
            "method",
            "enabled",
            "length",
            "seed",
            "watermark_device",
            "n_runs",
        },
        "pdw": {
            "method",
            "enabled",
            "sk_path",
            "pk_path",
            "params_path",
            "signature_segment_length",
            "bit_size",
            "message_length",
            "max_planted_errors",
            "seed",
        },
        "upv": {
            "method",
            "enabled",
            "detector_dir",
            "window_size",
            "delta",
            "gamma",
            "bit_number",
            "layers",
            "beam_size",
        },
    }
    if method not in allowed:
        raise PlanValidationError(
            "watermark.method must be one of: "
            + ", ".join(sorted(allowed))
        )
    unknown = sorted(set(value) - allowed[method])
    if unknown:
        raise PlanValidationError(
            f"{method} watermark has unknown settings: {', '.join(unknown)}"
        )
    resolved = dict(value)
    if not isinstance(value.get("enabled"), bool):
        raise PlanValidationError(
            f"{method} watermark requires an explicit boolean enabled setting"
        )
    if method == "none" and value["enabled"]:
        raise PlanValidationError(
            "watermark.method=none requires enabled=false"
        )
    if method == "vow":
        required = {"window_size", "delta", "gamma", "server_seed_path"}
        missing = sorted(required - set(value))
        if missing:
            raise PlanValidationError(
                f"vow watermark is missing settings: {', '.join(missing)}"
            )
        if value["window_size"] <= 0 or value["delta"] <= 0:
            raise PlanValidationError(
                "vow window_size and delta must be positive"
            )
        if not 0 < value["gamma"] < 1:
            raise PlanValidationError("vow gamma must be between zero and one")
        path, digest = _file(
            value["server_seed_path"],
            repository,
            label="VOW server seed",
        )
        resolved["server_seed_path"] = path
        resolved["server_seed_sha256"] = digest
        resolved["naive_baseline"] = value.get("naive_baseline", False)
    elif method in {"lefthash", "selfhash"}:
        required = {"delta", "gamma"}
        missing = sorted(required - set(value))
        if missing:
            raise PlanValidationError(
                f"{method} watermark is missing settings: "
                + ", ".join(missing)
            )
        if value["delta"] <= 0 or not 0 < value["gamma"] < 1:
            raise PlanValidationError(
                f"{method} delta must be positive and gamma between zero and one"
            )
    elif method == "rdf":
        required = {"length", "seed", "watermark_device", "n_runs"}
        missing = sorted(required - set(value))
        if missing:
            raise PlanValidationError(
                "rdf watermark is missing settings: " + ", ".join(missing)
            )
        if value["length"] <= 0 or value["n_runs"] <= 0:
            raise PlanValidationError(
                "rdf length and n_runs must be positive"
            )
        if value["watermark_device"] not in {"cpu", "gpu", "cuda"}:
            raise PlanValidationError(
                "rdf watermark_device must be cpu, gpu, or cuda"
            )
    elif method == "pdw":
        required = {
            "sk_path",
            "pk_path",
            "params_path",
            "signature_segment_length",
            "bit_size",
            "message_length",
            "max_planted_errors",
            "seed",
        }
        missing = sorted(required - set(value))
        if missing:
            raise PlanValidationError(
                "pdw watermark is missing settings: " + ", ".join(missing)
            )
        positive = (
            "signature_segment_length",
            "bit_size",
            "message_length",
        )
        if any(value[field] <= 0 for field in positive):
            raise PlanValidationError(
                "pdw segment, bit, and message lengths must be positive"
            )
        for field in ("sk_path", "pk_path", "params_path"):
            path, digest = _file(
                value[field], repository, label=f"PDW {field}"
            )
            resolved[field] = path
            resolved[f"{field}_sha256"] = digest
    elif method == "upv":
        required = {
            "detector_dir",
            "window_size",
            "delta",
            "gamma",
            "bit_number",
            "layers",
            "beam_size",
        }
        missing = sorted(required - set(value))
        if missing:
            raise PlanValidationError(
                "upv watermark is missing settings: " + ", ".join(missing)
            )
        if (
            value["window_size"] <= 0
            or value["delta"] <= 0
            or not 0 < value["gamma"] < 1
        ):
            raise PlanValidationError(
                "upv window_size/delta must be positive and gamma between "
                "zero and one"
            )
        directory = resolve_path(value["detector_dir"], repository)
        if not directory.is_dir():
            raise ResolutionError(
                f"UPV detector directory does not exist: {directory}"
            )
        resolved["detector_dir"] = str(directory)
        for name in ("combine_model.pt", "private_detector.pt"):
            path = directory / name
            if path.exists():
                resolved[f"{name}_sha256"] = sha256_file(path)
    return resolved


def validated_seed(watermark: JsonObject) -> bytes:
    path = Path(watermark["server_seed_path"])
    if sha256_file(path) != watermark["server_seed_sha256"]:
        raise ResolutionError(f"VOW server seed changed: {path}")
    return get_server_seed(path)


def generator_for(
    watermark: JsonObject, model: Any, tokenizer: Any
) -> tuple[WatermarkAdapter, bool]:
    method = watermark["method"]
    if method == "none" or not watermark["enabled"]:
        adapter = WatermarkAdapter()
        adapter.model = model
        adapter.tokenizer = tokenizer
        return adapter, True
    if method == "vow":
        return (
            VOWAdapter(
                model=model,
                tokenizer=tokenizer,
                window_size=watermark["window_size"],
                delta=watermark["delta"],
                gamma=watermark["gamma"],
                seed=validated_seed(watermark),
                naive_baseline=watermark["naive_baseline"],
            ),
            False,
        )
    if method in {"lefthash", "selfhash"}:
        return (
            KGWAdapter(
                model=model,
                tokenizer=tokenizer,
                delta=watermark["delta"],
                gamma=watermark["gamma"],
                seeding_scheme=method,
            ),
            False,
        )
    if method == "rdf":
        return (
            RDFAdapter(
                model=model,
                tokenizer=tokenizer,
                watermark_sequence_length=watermark["length"],
                seed=watermark["seed"],
                watermark_sequence_device=watermark["watermark_device"],
            ),
            False,
        )
    if method == "pdw":
        return (
            PDWAdapter(
                model=model,
                tokenizer=tokenizer,
                sk_path=watermark["sk_path"],
                pk_path=watermark["pk_path"],
                params_path=watermark["params_path"],
                signature_segment_length=watermark[
                    "signature_segment_length"
                ],
                bit_size=watermark["bit_size"],
                message_length=watermark["message_length"],
                max_planted_errors=watermark["max_planted_errors"],
                seed=watermark["seed"],
                timing=False,
            ),
            False,
        )
    if method == "upv":
        return (
            UPVAdapter(
                model,
                tokenizer,
                watermark["detector_dir"],
                window_size=watermark["window_size"],
                delta=watermark["delta"],
                gamma=watermark["gamma"],
                bit_number=watermark["bit_number"],
                layers=watermark["layers"],
                beam_size=watermark["beam_size"],
            ),
            False,
        )
    raise AssertionError(method)


def detector_for(
    watermark: JsonObject,
    tokenizer: Any,
    *,
    device: str,
) -> tuple[Any, JsonObject]:
    method = watermark["method"]
    selected_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    if selected_device == "auto":
        selected_device = "cpu"
    if method == "vow":
        detector = VOWDetector(
            tokenizer,
            seed=validated_seed(watermark),
            gamma=watermark["gamma"],
            window_size=watermark["window_size"],
        )
        return detector, {}
    if method in {"lefthash", "selfhash"}:
        return (
            KGWDetector(
                vocab=list(tokenizer.get_vocab().values()),
                gamma=watermark["gamma"],
                seeding_scheme=method,
                device=selected_device,
                tokenizer=tokenizer,
                normalizers=[],
                ignore_repeated_ngrams=True,
            ),
            {},
        )
    if method == "rdf":
        return (
            RDFDetector(
                tokenizer,
                length=watermark["length"],
                seed=watermark["seed"],
                n_runs=watermark["n_runs"],
            ),
            {},
        )
    if method == "pdw":
        return (
            PDWDetector(
                pk_path=watermark["pk_path"],
                params_path=watermark["params_path"],
                signature_segment_length=watermark[
                    "signature_segment_length"
                ],
                bit_size=watermark["bit_size"],
                message_length=watermark["message_length"],
                max_planted_errors=watermark["max_planted_errors"],
            ),
            {},
        )
    if method == "upv":
        return (
            UPVDetector(
                tokenizer,
                watermark["detector_dir"],
                window_size=watermark["window_size"],
                bits_num=watermark["bit_number"],
                gamma=watermark["gamma"],
            ),
            {},
        )
    raise PlanValidationError(
        "a detection stage cannot derive a detector from watermark.method=none"
    )
