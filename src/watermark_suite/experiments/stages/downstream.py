from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor

import torch
from datasets import DownloadConfig, load_dataset
from human_eval.data import read_problems
from human_eval.execution import check_correctness

from watermark_suite.evaluation.gsm8k import (
    STOP_STRINGS as GSM8K_STOP_STRINGS,
)
from watermark_suite.evaluation.gsm8k import (
    build_gsm8k_prompt,
    extract_gsm8k_answer,
)
from watermark_suite.evaluation.humaneval import (
    PROMPT_TEMPLATE as HUMANEVAL_PROMPT_TEMPLATE,
)
from watermark_suite.evaluation.humaneval import cleanup_code_official

from ..adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from ..errors import PlanValidationError, ResolutionError
from ..identity import derived_seed, identity_for
from ..models import ArtifactRef, JsonObject, WorkItem, WorkResult
from .common import (
    batches,
    require,
    resolve_model,
    validate_execution_settings,
)
from .watermarks import generator_for, resolve_watermark


_EXPECTED_SAMPLE_NUM = {"gsm8k": 1319, "humaneval": 164}


def _load_task_samples(task: str) -> list[JsonObject]:
    if task == "gsm8k":
        try:
            dataset = load_dataset(
                "openai/gsm8k",
                "main",
                split="test",
                download_config=DownloadConfig(local_files_only=True),
            )
        except Exception as error:
            raise ResolutionError(
                "GSM8K is not available in the local Hugging Face cache; "
                "Experiment Plan checking never downloads datasets"
            ) from error
        return [
            {
                "sample_id": f"gsm8k:{index}",
                "question": item["question"],
                "answer": item["answer"],
            }
            for index, item in enumerate(dataset)
        ]
    if task == "humaneval":
        return [
            {
                "sample_id": f"humaneval:{task_id}",
                "task_id": task_id,
                "problem": problem,
            }
            for task_id, problem in read_problems().items()
        ]
    raise AssertionError(task)


def resolve_task_dataset(task: str) -> JsonObject:
    samples = _load_task_samples(task)
    expected = _EXPECTED_SAMPLE_NUM[task]
    if len(samples) != expected:
        raise PlanValidationError(
            f"{task} must contain the full official {expected}-sample "
            f"evaluation set, found {len(samples)}"
        )
    return {
        "name": "openai/gsm8k" if task == "gsm8k" else "openai/human-eval",
        "config": "main" if task == "gsm8k" else None,
        "split": "test",
        "sample_num": len(samples),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "snapshot": identity_for(samples, prefix="dataset"),
    }


class DownstreamStageAdapter:
    kind = "downstream"
    revision = "downstream-v1"
    accepted_settings = {
        "task",
        "model",
        "batch_size",
        "seed",
        "max_new_tokens",
        "num_shots",
        "watermark",
        "evaluation_workers",
        "execution_timeout",
        "device",
        "dtype",
    }
    _required = accepted_settings

    def resolve(
        self, settings: JsonObject, context: ResolutionContext
    ) -> ResolvedStageDefinition:
        require(settings, self._required, kind=self.kind)
        validate_execution_settings(settings)
        task = settings["task"]
        if task not in _EXPECTED_SAMPLE_NUM:
            raise PlanValidationError("task must be gsm8k or humaneval")
        for field in (
            "batch_size",
            "max_new_tokens",
            "evaluation_workers",
        ):
            if not isinstance(settings[field], int) or settings[field] <= 0:
                raise PlanValidationError(f"{field} must be a positive integer")
        if not isinstance(settings["seed"], int):
            raise PlanValidationError("seed must be an integer")
        if (
            not isinstance(settings["execution_timeout"], (int, float))
            or settings["execution_timeout"] <= 0
        ):
            raise PlanValidationError("execution_timeout must be positive")
        expected_shots = 4 if task == "gsm8k" else 0
        if settings["num_shots"] != expected_shots:
            raise PlanValidationError(
                f"{task} requires num_shots={expected_shots}"
            )

        model = resolve_model(
            settings["model"],
            models=context.models,
            repository=context.repository,
        )
        watermark = resolve_watermark(
            settings["watermark"], context.repository
        )
        dataset = resolve_task_dataset(task)
        task_evaluation: JsonObject = {
            "num_shots": settings["num_shots"],
            "prompt_revision": (
                "gsm8k-official-four-shot-v1"
                if task == "gsm8k"
                else "humaneval-sft-zero-shot-v1"
            ),
        }
        if task == "gsm8k":
            task_evaluation["stop_strings"] = list(GSM8K_STOP_STRINGS)
        else:
            task_evaluation["execution_timeout"] = float(
                settings["execution_timeout"]
            )
            task_evaluation["cleanup_revision"] = "humaneval-official-v1"
        semantic = {
            "task": task,
            "model": model,
            "dataset": dataset,
            "batch_size": settings["batch_size"],
            "seed": settings["seed"],
            "max_new_tokens": settings["max_new_tokens"],
            "decoding": {"do_sample": False, "num_beams": 1},
            "task_evaluation": task_evaluation,
            "watermark": watermark,
        }
        execution = {
            "device": settings["device"],
            "dtype": settings["dtype"],
            "evaluation_workers": settings["evaluation_workers"],
        }
        return ResolvedStageDefinition(
            settings={**settings, "model": model},
            semantic_settings=semantic,
            execution_settings=execution,
            artifact_schema_revision=f"{task}-evaluation-v1",
            resource_key=(
                f"causal:{model['checkpoint']}@{model['revision']}:"
                f"{settings['device']}:{settings['dtype']}"
            ),
        )

    def bind_inputs(
        self,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
    ) -> ResolvedStageDefinition:
        if inputs:
            raise PlanValidationError(
                "downstream does not accept an input Artifact"
            )
        return definition

    def prepare(
        self, context: StageExecutionContext
    ) -> "_DownstreamExecution":
        return _DownstreamExecution(context)


class _DownstreamExecution:
    def __init__(self, context: StageExecutionContext) -> None:
        self.context = context
        semantic = context.semantic_settings
        self.model, self.tokenizer = context.runtime.get(
            semantic["model"],
            device=context.execution_settings["device"],
            dtype=context.execution_settings["dtype"],
            padding_side="left",
        )
        self.watermarker, self.no_watermark = generator_for(
            semantic["watermark"], self.model, self.tokenizer
        )
        self.samples = _load_task_samples(semantic["task"])
        dataset = semantic["dataset"]
        actual_ids = [sample["sample_id"] for sample in self.samples]
        if (
            actual_ids != dataset["sample_ids"]
            or identity_for(self.samples, prefix="dataset")
            != dataset["snapshot"]
        ):
            raise ResolutionError(
                f"{semantic['task']} changed after Experiment Plan resolution"
            )

    def work_items(self) -> list[WorkItem]:
        result = []
        for ordinal, batch in enumerate(
            batches(
                self.samples,
                self.context.semantic_settings["batch_size"],
            )
        ):
            sample_ids = tuple(item["sample_id"] for item in batch)
            result.append(
                WorkItem(
                    identity=identity_for(
                        {"ordinal": ordinal, "sample_ids": sample_ids},
                        prefix="batch",
                    ),
                    ordinal=ordinal,
                    sample_identities=sample_ids,
                    payload=batch,
                )
            )
        return result

    def _prompts(self, samples: list[JsonObject]) -> list[str]:
        semantic = self.context.semantic_settings
        if semantic["task"] == "gsm8k":
            return [
                build_gsm8k_prompt(
                    sample["question"],
                    semantic["task_evaluation"]["num_shots"],
                )
                for sample in samples
            ]
        return [
            HUMANEVAL_PROMPT_TEMPLATE.format(
                prompt=sample["problem"]["prompt"].strip()
            )
            for sample in samples
        ]

    def execute(self, item: WorkItem) -> WorkResult:
        semantic = self.context.semantic_settings
        prompts = self._prompts(item.payload)
        stop_strings = semantic["task_evaluation"].get("stop_strings")
        with torch.inference_mode():
            texts = self.watermarker(
                prompts=prompts,
                max_new_tokens=semantic["max_new_tokens"],
                stop_strings=stop_strings,
                seed=derived_seed(semantic["seed"], item.identity),
                do_sample=False,
                num_beams=1,
                top_p=None,
                top_k=None,
                temperature=None,
                pad_token_id=self.tokenizer.eos_token_id,
                no_watermark=self.no_watermark,
            )
        if not isinstance(texts, list):
            texts = [texts]
        if semantic["task"] == "gsm8k":
            records = self._gsm8k_records(item.payload, prompts, texts)
        else:
            records = self._humaneval_records(item.payload, prompts, texts)
        return WorkResult(records=tuple(records))

    def _gsm8k_records(
        self,
        samples: list[JsonObject],
        prompts: list[str],
        texts: list[str],
    ) -> list[JsonObject]:
        records = []
        for sample, prompt, text in zip(samples, prompts, texts):
            prediction = extract_gsm8k_answer(text)
            ground_truth = extract_gsm8k_answer(sample["answer"])
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "question": sample["question"],
                    "prompt_text": prompt,
                    "generated_text": text,
                    "generated_token_num": len(
                        self.tokenizer.encode(
                            text, add_special_tokens=False
                        )
                    ),
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "is_correct": prediction == ground_truth,
                }
            )
        return records

    def _humaneval_records(
        self,
        samples: list[JsonObject],
        prompts: list[str],
        texts: list[str],
    ) -> list[JsonObject]:
        completions = [cleanup_code_official(text) for text in texts]
        timeout = self.context.semantic_settings["task_evaluation"][
            "execution_timeout"
        ]

        def evaluate(arguments: tuple[JsonObject, str]) -> JsonObject:
            sample, completion = arguments
            return check_correctness(
                sample["problem"], completion, timeout, completion_id=0
            )

        workers = self.context.execution_settings["evaluation_workers"]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            evaluations = list(
                executor.map(evaluate, zip(samples, completions))
            )
        records = []
        for sample, prompt, text, completion, evaluation in zip(
            samples, prompts, texts, completions, evaluations
        ):
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "task_id": sample["task_id"],
                    "prompt_text": prompt,
                    "generated_text": text,
                    "generated_token_num": len(
                        self.tokenizer.encode(
                            text, add_special_tokens=False
                        )
                    ),
                    "completion": completion,
                    "test_result": evaluation["result"],
                    "passed": bool(evaluation["passed"]),
                }
            )
        return records

    def summarize(self, records: list[JsonObject]) -> JsonObject:
        lengths = [record["generated_token_num"] for record in records]
        summary: JsonObject = {
            "task": self.context.semantic_settings["task"],
            "sample_num": len(records),
            "watermark": self.context.semantic_settings["watermark"],
            "generated_token_num": sum(lengths),
            "mean_generated_token_num": (
                statistics.mean(lengths) if lengths else 0.0
            ),
        }
        if self.context.semantic_settings["task"] == "gsm8k":
            correct = sum(bool(record["is_correct"]) for record in records)
            summary.update(
                {
                    "correct_num": correct,
                    "accuracy": correct / len(records) if records else 0.0,
                    "accuracy_percent": (
                        100.0 * correct / len(records) if records else 0.0
                    ),
                    "num_shots": 4,
                }
            )
        else:
            passed = sum(bool(record["passed"]) for record in records)
            summary.update(
                {
                    "passed_num": passed,
                    "pass@1": passed / len(records) if records else 0.0,
                    "pass@1_percent": (
                        100.0 * passed / len(records) if records else 0.0
                    ),
                    "num_shots": 0,
                }
            )
        return summary

    def close(self) -> None:
        pass
