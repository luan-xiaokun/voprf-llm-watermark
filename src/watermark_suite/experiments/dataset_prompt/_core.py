from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...evaluation.gsm8k import (
    FEW_SHOT_EXAMPLES,
    STOP_STRINGS as GSM8K_STOP_STRINGS,
)
from ...evaluation.gsm8k import build_gsm8k_prompt
from ...evaluation.humaneval import PROMPT_TEMPLATE as HUMANEVAL_TEMPLATE
from ..errors import PlanValidationError, ResolutionError
from ..identity import directory_snapshot, identity_for, sha256_file
from ..models import JsonObject


DATASET_PROMPT_REVISION = "dataset-prompt-v1"

_ELI5_SYSTEM_MESSAGE = (
    "Explain the following question in about 500 words like I'm 5 years old. "
    "Use very simple language, short sentences, and analogies a child can "
    "understand."
)


def _policy(
    name: str,
    revision: str,
    parameters: JsonObject,
) -> JsonObject:
    document = {
        "name": name,
        "revision": revision,
        "parameters": parameters,
    }
    return {
        **document,
        "identity": identity_for(document, prefix="prompt-policy"),
    }


@dataclass(frozen=True)
class PromptSample:
    sample_id: str
    source_sample_id: str
    repetition: int
    source_prompt: str
    model_prompt: str
    payload: JsonObject


@dataclass(frozen=True)
class MaterializedPromptPopulation:
    samples: tuple[PromptSample, ...]

    def repeated(self, repetitions: int) -> "MaterializedPromptPopulation":
        if repetitions <= 0:
            raise PlanValidationError("repetitions must be positive")
        expanded = []
        for sample in self.samples:
            for repetition in range(repetitions):
                expanded.append(
                    PromptSample(
                        sample_id=(
                            sample.sample_id
                            if repetitions == 1
                            else (
                                f"{sample.sample_id}:repeat:{repetition}"
                            )
                        ),
                        source_sample_id=sample.sample_id,
                        repetition=repetition,
                        source_prompt=sample.source_prompt,
                        model_prompt=sample.model_prompt,
                        payload=sample.payload,
                    )
                )
        return MaterializedPromptPopulation(tuple(expanded))


@dataclass(frozen=True)
class ResolvedPromptPopulation:
    kind: str
    dataset: JsonObject
    source: JsonObject
    prompt_policy: JsonObject
    generation: JsonObject

    @property
    def sample_num(self) -> int:
        return int(self.dataset["selection"]["count"])

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self.dataset["selection"]["sample_ids"])

    def to_dict(self) -> JsonObject:
        return copy.deepcopy(
            {
                "revision": DATASET_PROMPT_REVISION,
                "kind": self.kind,
                "dataset": self.dataset,
                "source": self.source,
                "prompt_policy": self.prompt_policy,
                "generation": self.generation,
            }
        )

    @classmethod
    def from_dict(cls, value: JsonObject) -> "ResolvedPromptPopulation":
        if value.get("revision") != DATASET_PROMPT_REVISION:
            raise PlanValidationError(
                "unsupported Dataset & Prompt revision "
                f"{value.get('revision')!r}"
            )
        required = {
            "revision",
            "kind",
            "dataset",
            "source",
            "prompt_policy",
            "generation",
        }
        if set(value) != required:
            raise PlanValidationError(
                "Resolved Prompt Population fields disagree with "
                f"{DATASET_PROMPT_REVISION}"
            )
        for field in ("dataset", "source", "prompt_policy", "generation"):
            if not isinstance(value[field], dict):
                raise PlanValidationError(
                    f"Resolved Prompt Population {field} must be a mapping"
                )
        policy = value["prompt_policy"]
        expected_policy = _policy(
            str(policy.get("name")),
            str(policy.get("revision")),
            dict(policy.get("parameters", {})),
        )
        if policy != expected_policy:
            raise PlanValidationError(
                "Prompt Policy identity does not match its content"
            )
        selection = value["dataset"].get("selection")
        if not isinstance(selection, dict):
            raise PlanValidationError(
                "Resolved Prompt Population lacks Sample selection"
            )
        sample_ids = selection.get("sample_ids")
        count = selection.get("count")
        if (
            not isinstance(count, int)
            or count <= 0
            or not isinstance(sample_ids, list)
            or len(sample_ids) != count
            or not all(isinstance(item, str) and item for item in sample_ids)
            or len(sample_ids) != len(set(sample_ids))
        ):
            raise PlanValidationError(
                "Resolved Prompt Population has invalid Sample identities"
            )
        return cls(
            kind=str(value["kind"]),
            dataset=copy.deepcopy(value["dataset"]),
            source=copy.deepcopy(value["source"]),
            prompt_policy=copy.deepcopy(policy),
            generation=copy.deepcopy(value["generation"]),
        )


class _DatasetPromptAdapter:
    kind: str

    def resolve(
        self,
        raw: JsonObject,
        *,
        alias: str | None,
        repository: Path,
        sample_num: int | None,
    ) -> ResolvedPromptPopulation:
        raise NotImplementedError

    def load(
        self,
        source: JsonObject,
        *,
        limit: int | None = None,
    ) -> list[JsonObject]:
        raise NotImplementedError

    def snapshot(
        self,
        source: JsonObject,
        records: list[JsonObject],
    ) -> str:
        raise NotImplementedError

    def sample_identity(
        self,
        record: JsonObject,
        source: JsonObject,
    ) -> str:
        raise NotImplementedError

    def render(
        self,
        record: JsonObject,
        source: JsonObject,
        tokenizer: Any | None,
    ) -> tuple[str, str]:
        raise NotImplementedError


def _resolve_path(value: str, repository: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository / path
    return path.resolve()


def _jsonl_records(
    path: Path,
    *,
    limit: int | None = None,
) -> list[JsonObject]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if limit is not None and len(records) >= limit:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ResolutionError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error
            if not isinstance(record, dict):
                raise ResolutionError(
                    f"{path}:{line_number}: record must be an object"
                )
            records.append(record)
    return records


class _TextDatasetAdapter(_DatasetPromptAdapter):
    default_format: str
    default_path: str
    default_prompt_field: str
    default_id_field: str | None

    def resolve(
        self,
        raw: JsonObject,
        *,
        alias: str | None,
        repository: Path,
        sample_num: int | None,
    ) -> ResolvedPromptPopulation:
        if not isinstance(sample_num, int) or sample_num <= 0:
            raise PlanValidationError(
                f"{self.kind} requires a positive sample_num"
            )
        allowed = {
            "kind",
            "format",
            "path",
            "split",
            "prompt_field",
            "sample_id_field",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PlanValidationError(
                f"dataset has unknown fields: {', '.join(unknown)}"
            )
        data_format = raw.get("format", self.default_format)
        if data_format not in {"jsonl", "huggingface"}:
            raise PlanValidationError(
                "dataset format must be jsonl or huggingface"
            )
        path_value = raw.get("path", self.default_path)
        if not isinstance(path_value, str):
            raise PlanValidationError("dataset needs a path")
        path = _resolve_path(path_value, repository)
        if not path.exists():
            raise ResolutionError(f"dataset path does not exist: {path}")
        source = {
            "alias": alias,
            "format": data_format,
            "path": str(path),
            "split": raw.get("split", "train"),
            "prompt_field": raw.get(
                "prompt_field", self.default_prompt_field
            ),
            "sample_id_field": raw.get(
                "sample_id_field", self.default_id_field
            ),
        }
        records = self.load(source, limit=sample_num)
        if len(records) != sample_num:
            raise PlanValidationError(
                f"dataset contains {len(records)} records, fewer than "
                f"sample_num={sample_num}"
            )
        snapshot = self.snapshot(source, records)
        sample_ids = [
            self.sample_identity(record, source) for record in records
        ]
        _require_unique(sample_ids, label=f"{self.kind} Sample")
        dataset = {
            "kind": self.kind,
            "split": source["split"],
            "snapshot": snapshot,
            "selection": {
                "mode": "first",
                "count": sample_num,
                "sample_ids": sample_ids,
            },
        }
        return ResolvedPromptPopulation(
            kind=self.kind,
            dataset=dataset,
            source=source,
            prompt_policy=self.prompt_policy(source),
            generation={},
        )

    def load(
        self,
        source: JsonObject,
        *,
        limit: int | None = None,
    ) -> list[JsonObject]:
        if source["format"] == "jsonl":
            return _jsonl_records(Path(source["path"]), limit=limit)
        from datasets import load_dataset

        dataset = load_dataset(source["path"], split=source["split"])
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        return [dict(item) for item in dataset]

    def snapshot(
        self,
        source: JsonObject,
        records: list[JsonObject],
    ) -> str:
        del records
        path = Path(source["path"])
        return sha256_file(path) if path.is_file() else directory_snapshot(path)

    def sample_identity(
        self,
        record: JsonObject,
        source: JsonObject,
    ) -> str:
        id_field = source["sample_id_field"]
        if id_field and id_field in record:
            return f"{self.kind}:{record[id_field]}"
        content = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return f"{self.kind}:{digest}"

    def prompt_policy(self, source: JsonObject) -> JsonObject:
        raise NotImplementedError


class _C4Adapter(_TextDatasetAdapter):
    kind = "c4"
    default_format = "jsonl"
    default_path = "data/c4_realnewslike_subset_1000.jsonl"
    default_prompt_field = "prompt_text"
    default_id_field = "original_index"

    def prompt_policy(self, source: JsonObject) -> JsonObject:
        return _policy(
            "c4-plain-text",
            "c4-plain-text-v1",
            {"source_field": source["prompt_field"]},
        )

    def render(
        self,
        record: JsonObject,
        source: JsonObject,
        tokenizer: Any | None,
    ) -> tuple[str, str]:
        del tokenizer
        prompt = str(record[source["prompt_field"]])
        return prompt, prompt


class _Eli5Adapter(_TextDatasetAdapter):
    kind = "eli5"
    default_format = "huggingface"
    default_path = "data/eli5"
    default_prompt_field = "question"
    default_id_field = None

    def prompt_policy(self, source: JsonObject) -> JsonObject:
        instruction_identity = identity_for(
            {
                "system": _ELI5_SYSTEM_MESSAGE,
                "fallback": "system\\n\\nQuestion: {prompt}\\nAnswer:",
            },
            prefix="instruction",
        )
        return _policy(
            "eli5-chat-explanation",
            "eli5-chat-explanation-v1",
            {
                "source_field": source["prompt_field"],
                "instruction_identity": instruction_identity,
            },
        )

    def render(
        self,
        record: JsonObject,
        source: JsonObject,
        tokenizer: Any | None,
    ) -> tuple[str, str]:
        if tokenizer is None:
            raise PlanValidationError(
                "ELI5 prompt rendering requires a tokenizer"
            )
        source_prompt = str(record[source["prompt_field"]])
        if getattr(tokenizer, "chat_template", None) is not None:
            model_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": _ELI5_SYSTEM_MESSAGE},
                    {"role": "user", "content": source_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            model_prompt = (
                f"{_ELI5_SYSTEM_MESSAGE}\n\n"
                f"Question: {source_prompt}\nAnswer:"
            )
        return source_prompt, model_prompt


class _Gsm8kAdapter(_DatasetPromptAdapter):
    kind = "gsm8k"
    expected_sample_num = 1319

    def resolve(
        self,
        raw: JsonObject,
        *,
        alias: str | None,
        repository: Path,
        sample_num: int | None,
    ) -> ResolvedPromptPopulation:
        del alias, repository
        if set(raw) != {"kind"}:
            raise PlanValidationError(
                "GSM8K Prompt Population has no configurable dataset fields"
            )
        if sample_num not in {None, self.expected_sample_num}:
            raise PlanValidationError(
                f"gsm8k requires its full {self.expected_sample_num} Samples"
            )
        source = {
            "name": "openai/gsm8k",
            "config": "main",
            "split": "test",
        }
        records = self.load(source)
        _require_count(records, self.expected_sample_num, label="gsm8k")
        return self._resolved(source, records)

    def _resolved(
        self,
        source: JsonObject,
        records: list[JsonObject],
    ) -> ResolvedPromptPopulation:
        sample_ids = [
            self.sample_identity(record, source) for record in records
        ]
        _require_unique(sample_ids, label="gsm8k Sample")
        probe = build_gsm8k_prompt("__PROMPT_POLICY_PROBE__", 4)
        policy = _policy(
            "gsm8k-official-four-shot",
            "gsm8k-official-four-shot-v1",
            {
                "num_shots": 4,
                "template_identity": identity_for(
                    {
                        "probe": probe,
                        "examples": FEW_SHOT_EXAMPLES,
                    },
                    prefix="prompt-template",
                ),
            },
        )
        snapshot = self.snapshot(source, records)
        return ResolvedPromptPopulation(
            kind=self.kind,
            dataset={
                "kind": self.kind,
                "name": source["name"],
                "split": source["split"],
                "snapshot": snapshot,
                "selection": {
                    "mode": "full",
                    "count": len(records),
                    "sample_ids": sample_ids,
                },
            },
            source=source,
            prompt_policy=policy,
            generation={"stop_strings": list(GSM8K_STOP_STRINGS)},
        )

    def load(
        self,
        source: JsonObject,
        *,
        limit: int | None = None,
    ) -> list[JsonObject]:
        from datasets import DownloadConfig, load_dataset

        try:
            dataset = load_dataset(
                source["name"],
                source["config"],
                split=source["split"],
                download_config=DownloadConfig(local_files_only=True),
            )
        except Exception as error:
            raise ResolutionError(
                "GSM8K is not available in the local Hugging Face cache; "
                "Experiment Plan checking never downloads datasets"
            ) from error
        records = [
            {
                "_ordinal": index,
                "question": item["question"],
                "answer": item["answer"],
            }
            for index, item in enumerate(dataset)
        ]
        return records if limit is None else records[:limit]

    def snapshot(
        self,
        source: JsonObject,
        records: list[JsonObject],
    ) -> str:
        del source
        return identity_for(records, prefix="dataset")

    def sample_identity(
        self,
        record: JsonObject,
        source: JsonObject,
    ) -> str:
        del source
        return f"gsm8k:{record['_ordinal']}"

    def render(
        self,
        record: JsonObject,
        source: JsonObject,
        tokenizer: Any | None,
    ) -> tuple[str, str]:
        del source, tokenizer
        prompt = str(record["question"])
        return prompt, build_gsm8k_prompt(prompt, 4)


class _HumanEvalAdapter(_DatasetPromptAdapter):
    kind = "humaneval"
    expected_sample_num = 164

    def resolve(
        self,
        raw: JsonObject,
        *,
        alias: str | None,
        repository: Path,
        sample_num: int | None,
    ) -> ResolvedPromptPopulation:
        del alias, repository
        if set(raw) != {"kind"}:
            raise PlanValidationError(
                "HumanEval Prompt Population has no configurable dataset fields"
            )
        if sample_num not in {None, self.expected_sample_num}:
            raise PlanValidationError(
                "humaneval requires its full 164 Samples"
            )
        source = {
            "name": "openai/human-eval",
            "config": None,
            "split": "test",
        }
        records = self.load(source)
        _require_count(records, self.expected_sample_num, label="humaneval")
        sample_ids = [
            self.sample_identity(record, source) for record in records
        ]
        _require_unique(sample_ids, label="humaneval Sample")
        policy = _policy(
            "humaneval-sft-zero-shot",
            "humaneval-sft-zero-shot-v1",
            {
                "num_shots": 0,
                "template_identity": identity_for(
                    HUMANEVAL_TEMPLATE,
                    prefix="prompt-template",
                ),
            },
        )
        snapshot = self.snapshot(source, records)
        return ResolvedPromptPopulation(
            kind=self.kind,
            dataset={
                "kind": self.kind,
                "name": source["name"],
                "split": source["split"],
                "snapshot": snapshot,
                "selection": {
                    "mode": "full",
                    "count": len(records),
                    "sample_ids": sample_ids,
                },
            },
            source=source,
            prompt_policy=policy,
            generation={},
        )

    def load(
        self,
        source: JsonObject,
        *,
        limit: int | None = None,
    ) -> list[JsonObject]:
        del source
        from human_eval.data import read_problems

        records = [
            {
                "task_id": task_id,
                "problem": problem,
            }
            for task_id, problem in read_problems().items()
        ]
        return records if limit is None else records[:limit]

    def snapshot(
        self,
        source: JsonObject,
        records: list[JsonObject],
    ) -> str:
        del source
        return identity_for(records, prefix="dataset")

    def sample_identity(
        self,
        record: JsonObject,
        source: JsonObject,
    ) -> str:
        del source
        return f"humaneval:{record['task_id']}"

    def render(
        self,
        record: JsonObject,
        source: JsonObject,
        tokenizer: Any | None,
    ) -> tuple[str, str]:
        del source, tokenizer
        source_prompt = str(record["problem"]["prompt"]).strip()
        return (
            source_prompt,
            HUMANEVAL_TEMPLATE.format(prompt=source_prompt),
        )


def _require_unique(values: list[str], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise PlanValidationError(f"{label} identities must be unique")


def _require_count(
    values: list[JsonObject],
    expected: int,
    *,
    label: str,
) -> None:
    if len(values) != expected:
        raise PlanValidationError(
            f"{label} must contain {expected} Samples, found {len(values)}"
        )


class DatasetPromptRegistry:
    """Resolve and materialize stable Prompt Populations."""

    def __init__(
        self,
        adapters: tuple[_DatasetPromptAdapter, ...],
    ) -> None:
        self._adapters = {adapter.kind: adapter for adapter in adapters}

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def resolve(
        self,
        value: object,
        *,
        catalog: JsonObject,
        repository: Path,
        sample_num: int | None = None,
    ) -> ResolvedPromptPopulation:
        alias = (
            value
            if isinstance(value, str) and value in catalog
            else None
        )
        raw = catalog[value] if alias is not None else value
        if isinstance(raw, str):
            raw = {"kind": raw}
        if not isinstance(raw, dict):
            raise PlanValidationError(
                f"Prompt Population {value!r} must resolve to a mapping"
            )
        kind = raw.get("kind")
        try:
            adapter = self._adapters[kind]
        except (KeyError, TypeError) as error:
            raise PlanValidationError(
                f"unknown Prompt Population kind {kind!r}; expected one of "
                + ", ".join(self.kinds)
            ) from error
        return adapter.resolve(
            dict(raw),
            alias=alias,
            repository=repository,
            sample_num=sample_num,
        )

    def materialize(
        self,
        value: ResolvedPromptPopulation | JsonObject,
        *,
        tokenizer: Any | None = None,
    ) -> MaterializedPromptPopulation:
        resolved = (
            value
            if isinstance(value, ResolvedPromptPopulation)
            else ResolvedPromptPopulation.from_dict(value)
        )
        try:
            adapter = self._adapters[resolved.kind]
        except KeyError as error:
            raise PlanValidationError(
                f"unsupported materialized Prompt Population "
                f"{resolved.kind!r}"
            ) from error
        records = adapter.load(
            resolved.source,
            limit=resolved.sample_num,
        )
        count = resolved.sample_num
        selected = records[:count]
        if len(selected) != count:
            raise ResolutionError(
                f"Prompt Population now has {len(selected)} Samples; "
                f"expected {count}"
            )
        snapshot = adapter.snapshot(resolved.source, selected)
        if snapshot != resolved.dataset["snapshot"]:
            raise ResolutionError(
                "Prompt Population changed after Plan resolution"
            )
        actual_ids = [
            adapter.sample_identity(record, resolved.source)
            for record in selected
        ]
        if tuple(actual_ids) != resolved.sample_ids:
            raise ResolutionError(
                "Prompt Population Sample manifest no longer matches"
            )
        samples = []
        for sample_id, record in zip(actual_ids, selected):
            source_prompt, model_prompt = adapter.render(
                record,
                resolved.source,
                tokenizer,
            )
            samples.append(
                PromptSample(
                    sample_id=sample_id,
                    source_sample_id=sample_id,
                    repetition=0,
                    source_prompt=source_prompt,
                    model_prompt=model_prompt,
                    payload=dict(record),
                )
            )
        return MaterializedPromptPopulation(tuple(samples))


DATASET_PROMPTS = DatasetPromptRegistry(
    (
        _C4Adapter(),
        _Eli5Adapter(),
        _Gsm8kAdapter(),
        _HumanEvalAdapter(),
    )
)
