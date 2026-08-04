from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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

from .errors import PlanValidationError, ResolutionError
from .identity import identity_for, sha256_file
from .models import JsonObject


GeneratorFactory = Callable[[JsonObject, Any, Any], WatermarkAdapter]
DetectorFactory = Callable[[JsonObject, Any, str], Any]
Resolver = Callable[[JsonObject, Path], JsonObject]
Verifier = Callable[[JsonObject], None]


@dataclass(frozen=True)
class _SchemeDefinition:
    accepted_fields: frozenset[str]
    resolve: Resolver
    generator: GeneratorFactory | None
    detector: DetectorFactory | None
    verify: Verifier


def _path(value: str, repository: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository / path
    return path.resolve()


def _file(value: str, repository: Path, *, label: str) -> tuple[str, str]:
    path = _path(value, repository)
    if not path.is_file():
        raise ResolutionError(f"{label} does not exist: {path}")
    return str(path), sha256_file(path)


def _verify_file(path: Path, digest: str, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != digest:
        raise ResolutionError(
            f"{label} changed after Plan resolution: {path}"
        )


def _resolve_none(value: JsonObject, repository: Path) -> JsonObject:
    del repository
    if value["enabled"]:
        raise PlanValidationError(
            "watermark.method=none requires enabled=false"
        )
    return value


def _resolve_vow(value: JsonObject, repository: Path) -> JsonObject:
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
    return {
        **value,
        "server_seed_path": path,
        "server_seed_sha256": digest,
        "naive_baseline": value.get("naive_baseline", False),
    }


def _resolve_kgw(value: JsonObject, repository: Path) -> JsonObject:
    del repository
    missing = sorted({"delta", "gamma"} - set(value))
    if missing:
        raise PlanValidationError(
            f"{value['method']} watermark is missing settings: "
            + ", ".join(missing)
        )
    if value["delta"] <= 0 or not 0 < value["gamma"] < 1:
        raise PlanValidationError(
            f"{value['method']} delta must be positive and gamma between "
            "zero and one"
        )
    return value


def _resolve_rdf(value: JsonObject, repository: Path) -> JsonObject:
    del repository
    required = {"length", "seed", "watermark_device", "n_runs"}
    missing = sorted(required - set(value))
    if missing:
        raise PlanValidationError(
            "rdf watermark is missing settings: " + ", ".join(missing)
        )
    if value["length"] <= 0 or value["n_runs"] <= 0:
        raise PlanValidationError("rdf length and n_runs must be positive")
    if value["watermark_device"] not in {"cpu", "gpu", "cuda"}:
        raise PlanValidationError(
            "rdf watermark_device must be cpu, gpu, or cuda"
        )
    return value


def _resolve_pdw(value: JsonObject, repository: Path) -> JsonObject:
    required = {
        "sk_path",
        "pk_path",
        "params_path",
        "signature_segment_length",
        "bit_size",
        "message_length",
        "max_planted_errors",
        "max_generation_attempts",
        "seed",
    }
    missing = sorted(required - set(value))
    if missing:
        raise PlanValidationError(
            "pdw watermark is missing settings: " + ", ".join(missing)
        )
    if any(
        value[field] <= 0
        for field in (
            "signature_segment_length",
            "bit_size",
            "message_length",
        )
    ):
        raise PlanValidationError(
            "pdw segment, bit, and message lengths must be positive"
        )
    if value["max_generation_attempts"] <= 0:
        raise PlanValidationError(
            "pdw max_generation_attempts must be positive"
        )
    resolved = {**value, "implementation_revision": "pdw-seeded-retry-v2"}
    for field in ("sk_path", "pk_path", "params_path"):
        path, digest = _file(
            value[field], repository, label=f"PDW {field}"
        )
        resolved[field] = path
        resolved[f"{field}_sha256"] = digest
    return resolved


def _resolve_upv(value: JsonObject, repository: Path) -> JsonObject:
    required = {
        "calibration_path",
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
    directory = _path(value["detector_dir"], repository)
    if not directory.is_dir():
        raise ResolutionError(
            f"UPV detector directory does not exist: {directory}"
        )
    resolved = {**value, "detector_dir": str(directory)}
    for name in ("combine_model.pt", "private_detector.pt"):
        path = directory / name
        if not path.is_file():
            raise ResolutionError(
                f"UPV detector material does not exist: {path}"
            )
        resolved[f"{name}_sha256"] = sha256_file(path)
    calibration_path, calibration_digest = _file(
        value["calibration_path"],
        repository,
        label="UPV classifier calibration",
    )
    try:
        calibration = json.loads(
            Path(calibration_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ResolutionError(
            f"UPV classifier calibration is invalid: {calibration_path}"
        ) from error
    if not isinstance(calibration, dict):
        raise PlanValidationError("UPV classifier calibration must be a mapping")
    if (
        calibration.get("schema_revision")
        != "upv-classifier-calibration-v1"
        or calibration.get("kind") != "classifier-threshold"
        or calibration.get("score_type") != "classifier_confidence"
        or calibration.get("decision_operator") != ">"
        or calibration.get("decision_threshold") != 0.5
    ):
        raise PlanValidationError(
            "UPV classifier calibration must declare the fixed "
            "classifier_confidence > 0.5 operating point"
        )
    empirical = calibration.get("empirical_fpr")
    if not isinstance(empirical, dict):
        raise PlanValidationError(
            "UPV classifier calibration lacks empirical_fpr"
        )
    try:
        positive_num = int(empirical["positive_num"])
        sample_num = int(empirical["sample_num"])
        rate = float(empirical["rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise PlanValidationError(
            "UPV empirical_fpr requires positive_num, sample_num, and rate"
        ) from error
    expected_rate = positive_num / sample_num if sample_num else -1.0
    if (
        sample_num <= 0
        or positive_num < 0
        or positive_num > sample_num
        or not math.isclose(rate, expected_rate, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise PlanValidationError(
            "UPV empirical_fpr rate disagrees with its exact counts"
        )
    if (
        calibration.get("private_detector_sha256")
        != resolved["private_detector.pt_sha256"]
    ):
        raise PlanValidationError(
            "UPV calibration was not measured with private_detector.pt"
        )
    if not isinstance(calibration.get("calibration"), dict):
        raise PlanValidationError(
            "UPV classifier calibration lacks calibration provenance"
        )
    resolved.update(
        {
            "calibration_path": calibration_path,
            "calibration_path_sha256": calibration_digest,
            "detection_operating_point": calibration,
        }
    )
    return resolved


def _verify_none_or_inline(value: JsonObject) -> None:
    del value


def _verify_vow(value: JsonObject) -> None:
    _verify_file(
        Path(value["server_seed_path"]),
        value["server_seed_sha256"],
        label="VOW server seed",
    )


def _verify_pdw(value: JsonObject) -> None:
    for field in ("sk_path", "pk_path", "params_path"):
        _verify_file(
            Path(value[field]),
            value[f"{field}_sha256"],
            label=f"PDW {field}",
        )


def _verify_upv(value: JsonObject) -> None:
    directory = Path(value["detector_dir"])
    for name in ("combine_model.pt", "private_detector.pt"):
        _verify_file(
            directory / name,
            value[f"{name}_sha256"],
            label=f"UPV {name}",
        )
    _verify_file(
        Path(value["calibration_path"]),
        value["calibration_path_sha256"],
        label="UPV classifier calibration",
    )


def _vow_seed(value: JsonObject) -> bytes:
    _verify_vow(value)
    return get_server_seed(Path(value["server_seed_path"]))


def _vow_generator(value: JsonObject, model: Any, tokenizer: Any) -> Any:
    return VOWAdapter(
        model=model,
        tokenizer=tokenizer,
        window_size=value["window_size"],
        delta=value["delta"],
        gamma=value["gamma"],
        seed=_vow_seed(value),
        naive_baseline=value["naive_baseline"],
    )


def _vow_detector(value: JsonObject, tokenizer: Any, device: str) -> Any:
    del device
    return VOWDetector(
        tokenizer,
        seed=_vow_seed(value),
        gamma=value["gamma"],
        window_size=value["window_size"],
    )


def _kgw_generator(value: JsonObject, model: Any, tokenizer: Any) -> Any:
    return KGWAdapter(
        model=model,
        tokenizer=tokenizer,
        delta=value["delta"],
        gamma=value["gamma"],
        seeding_scheme=value["method"],
    )


def _selected_device(device: str) -> str:
    selected = (
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    return "cpu" if selected == "auto" else selected


def _kgw_detector(value: JsonObject, tokenizer: Any, device: str) -> Any:
    return KGWDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=value["gamma"],
        seeding_scheme=value["method"],
        device=_selected_device(device),
        tokenizer=tokenizer,
        normalizers=[],
        ignore_repeated_ngrams=True,
    )


def _rdf_generator(value: JsonObject, model: Any, tokenizer: Any) -> Any:
    return RDFAdapter(
        model=model,
        tokenizer=tokenizer,
        watermark_sequence_length=value["length"],
        seed=value["seed"],
        watermark_sequence_device=value["watermark_device"],
    )


def _rdf_detector(value: JsonObject, tokenizer: Any, device: str) -> Any:
    del device
    return RDFDetector(
        tokenizer,
        length=value["length"],
        seed=value["seed"],
        n_runs=value["n_runs"],
    )


def _pdw_generator(value: JsonObject, model: Any, tokenizer: Any) -> Any:
    return PDWAdapter(
        model=model,
        tokenizer=tokenizer,
        sk_path=value["sk_path"],
        pk_path=value["pk_path"],
        params_path=value["params_path"],
        signature_segment_length=value["signature_segment_length"],
        bit_size=value["bit_size"],
        message_length=value["message_length"],
        max_planted_errors=value["max_planted_errors"],
        max_generation_attempts=value["max_generation_attempts"],
        seed=value["seed"],
        timing=False,
    )


def _pdw_detector(value: JsonObject, tokenizer: Any, device: str) -> Any:
    del tokenizer, device
    return PDWDetector(
        pk_path=value["pk_path"],
        params_path=value["params_path"],
        signature_segment_length=value["signature_segment_length"],
        bit_size=value["bit_size"],
        message_length=value["message_length"],
        max_planted_errors=value["max_planted_errors"],
    )


def _upv_generator(value: JsonObject, model: Any, tokenizer: Any) -> Any:
    return UPVAdapter(
        model,
        tokenizer,
        value["detector_dir"],
        window_size=value["window_size"],
        delta=value["delta"],
        gamma=value["gamma"],
        bit_number=value["bit_number"],
        layers=value["layers"],
        beam_size=value["beam_size"],
    )


def _upv_detector(value: JsonObject, tokenizer: Any, device: str) -> Any:
    del device
    return UPVDetector(
        tokenizer,
        value["detector_dir"],
        window_size=value["window_size"],
        bits_num=value["bit_number"],
        gamma=value["gamma"],
    )


class WatermarkSchemeRegistry:
    """Deep paired interface for watermark resolution and construction."""

    def __init__(self, definitions: dict[str, _SchemeDefinition]) -> None:
        self._definitions = dict(definitions)

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    def resolve(self, value: Any, repository: Path) -> JsonObject:
        if not isinstance(value, dict):
            raise PlanValidationError("watermark must be a mapping")
        method = value.get("method")
        if method not in self._definitions:
            raise PlanValidationError(
                "watermark.method must be one of: "
                + ", ".join(self.methods)
            )
        definition = self._definitions[method]
        unknown = sorted(set(value) - definition.accepted_fields)
        if unknown:
            raise PlanValidationError(
                f"{method} watermark has unknown settings: "
                + ", ".join(unknown)
            )
        if not isinstance(value.get("enabled"), bool):
            raise PlanValidationError(
                f"{method} watermark requires an explicit boolean enabled "
                "setting"
            )
        return definition.resolve(dict(value), repository)

    def verify(self, value: JsonObject) -> None:
        self._definition(value).verify(value)

    def generator(
        self,
        value: JsonObject,
        model: Any,
        tokenizer: Any,
    ) -> tuple[WatermarkAdapter, bool]:
        self.verify(value)
        if value["method"] == "none" or not value["enabled"]:
            adapter = WatermarkAdapter()
            adapter.model = model
            adapter.tokenizer = tokenizer
            return adapter, True
        factory = self._definition(value).generator
        if factory is None:
            raise PlanValidationError(
                f"{value['method']} has no generation adapter"
            )
        return factory(value, model, tokenizer), False

    def detector(
        self,
        value: JsonObject,
        tokenizer: Any,
        *,
        device: str,
    ) -> tuple[Any, JsonObject]:
        self.verify(value)
        factory = self._definition(value).detector
        if factory is None:
            raise PlanValidationError(
                "a detection stage cannot derive a detector from "
                f"watermark.method={value['method']}"
            )
        return factory(value, tokenizer, device), {}

    def vow_seed(self, value: JsonObject) -> bytes:
        if value.get("method") != "vow":
            raise PlanValidationError("VOW seed requested for another scheme")
        return _vow_seed(value)

    def analysis_dimensions(self, value: JsonObject) -> JsonObject:
        """Return stable scientific axes without local material paths."""

        method = value.get("method")
        fields = {
            "none": (),
            "vow": ("window_size", "gamma", "delta", "naive_baseline"),
            "lefthash": ("gamma", "delta"),
            "selfhash": ("gamma", "delta"),
            "rdf": ("length", "seed", "n_runs"),
            "pdw": (
                "signature_segment_length",
                "bit_size",
                "message_length",
                "max_planted_errors",
                "max_generation_attempts",
                "seed",
            ),
            "upv": (
                "window_size",
                "gamma",
                "delta",
                "bit_number",
                "layers",
                "beam_size",
            ),
        }
        if method not in fields:
            raise PlanValidationError(
                f"unknown resolved watermark method {method!r}"
            )
        parameters = {
            field: value[field]
            for field in fields[method]
            if field in value
        }
        document = {"method": method, "parameters": parameters}
        return {
            **document,
            "identity": identity_for(document, prefix="scheme"),
        }

    def _definition(self, value: JsonObject) -> _SchemeDefinition:
        method = value.get("method")
        try:
            return self._definitions[method]
        except (KeyError, TypeError) as error:
            raise PlanValidationError(
                f"unknown resolved watermark method {method!r}"
            ) from error


_COMMON = frozenset({"method", "enabled"})
WATERMARK_SCHEMES = WatermarkSchemeRegistry(
    {
        "none": _SchemeDefinition(
            _COMMON,
            _resolve_none,
            None,
            None,
            _verify_none_or_inline,
        ),
        "vow": _SchemeDefinition(
            _COMMON
            | {
                "window_size",
                "delta",
                "gamma",
                "server_seed_path",
                "naive_baseline",
            },
            _resolve_vow,
            _vow_generator,
            _vow_detector,
            _verify_vow,
        ),
        "lefthash": _SchemeDefinition(
            _COMMON | {"delta", "gamma"},
            _resolve_kgw,
            _kgw_generator,
            _kgw_detector,
            _verify_none_or_inline,
        ),
        "selfhash": _SchemeDefinition(
            _COMMON | {"delta", "gamma"},
            _resolve_kgw,
            _kgw_generator,
            _kgw_detector,
            _verify_none_or_inline,
        ),
        "rdf": _SchemeDefinition(
            _COMMON
            | {"length", "seed", "watermark_device", "n_runs"},
            _resolve_rdf,
            _rdf_generator,
            _rdf_detector,
            _verify_none_or_inline,
        ),
        "pdw": _SchemeDefinition(
            _COMMON
            | {
                "sk_path",
                "pk_path",
                "params_path",
                "signature_segment_length",
                "bit_size",
                "message_length",
                "max_planted_errors",
                "max_generation_attempts",
                "seed",
            },
            _resolve_pdw,
            _pdw_generator,
            _pdw_detector,
            _verify_pdw,
        ),
        "upv": _SchemeDefinition(
            _COMMON
            | {
                "calibration_path",
                "detector_dir",
                "window_size",
                "delta",
                "gamma",
                "bit_number",
                "layers",
                "beam_size",
            },
            _resolve_upv,
            _upv_generator,
            _upv_detector,
            _verify_upv,
        ),
    }
)
