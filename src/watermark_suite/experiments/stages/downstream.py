from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor

import torch
from human_eval.execution import check_correctness

from watermark_suite.evaluation.gsm8k import extract_gsm8k_answer
from watermark_suite.evaluation.humaneval import cleanup_code_official

from ..adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from ..dataset_prompt import DATASET_PROMPTS, PromptSample
from ..errors import PlanValidationError
from ..identity import derived_seed, identity_for
from ..models import ArtifactRef, JsonObject, WorkItem, WorkResult
from ..scheme_registry import WATERMARK_SCHEMES
from .common import (
    batches,
    require,
    resolve_model,
    validate_execution_settings,
)


_EXPECTED_SAMPLE_NUM = {"gsm8k": 1319, "humaneval": 164}


class DownstreamStageAdapter:
    kind = "downstream"
    revision = "downstream-v2"
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
        watermark = WATERMARK_SCHEMES.resolve(
            settings["watermark"], context.repository
        )
        prompt_population = DATASET_PROMPTS.resolve(
            {"kind": task},
            catalog={},
            repository=context.repository,
        )
        policy_shots = prompt_population.prompt_policy["parameters"][
            "num_shots"
        ]
        if settings["num_shots"] != policy_shots:
            raise PlanValidationError(
                f"{task} Prompt Policy requires num_shots={policy_shots}"
            )
        task_evaluation: JsonObject = {}
        if task == "humaneval":
            task_evaluation["execution_timeout"] = float(
                settings["execution_timeout"]
            )
            task_evaluation["cleanup_revision"] = "humaneval-official-v1"
        decoding: JsonObject = {"do_sample": False, "num_beams": 1}
        if watermark["method"] == "pdw":
            decoding["effective_sampling"] = "multinomial"
        semantic = {
            "task": task,
            "model": model,
            "prompt_population": prompt_population.to_dict(),
            "batch_size": settings["batch_size"],
            "seed": settings["seed"],
            "max_new_tokens": settings["max_new_tokens"],
            "decoding": decoding,
            "task_evaluation": task_evaluation,
            "watermark": watermark,
        }
        execution = {
            "device": settings["device"],
            "dtype": settings["dtype"],
            "evaluation_workers": settings["evaluation_workers"],
        }
        return ResolvedStageDefinition(
            settings={
                **settings,
                "model": model,
                "dataset": prompt_population.to_dict(),
            },
            semantic_settings=semantic,
            execution_settings=execution,
            artifact_schema_revision=f"{task}-evaluation-v2",
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
        self.watermarker, self.no_watermark = WATERMARK_SCHEMES.generator(
            semantic["watermark"], self.model, self.tokenizer
        )
        self.samples = list(
            DATASET_PROMPTS.materialize(
                semantic["prompt_population"],
                tokenizer=self.tokenizer,
            ).samples
        )

    def work_items(self) -> list[WorkItem]:
        result = []
        for ordinal, batch in enumerate(
            batches(
                self.samples,
                self.context.semantic_settings["batch_size"],
            )
        ):
            sample_ids = tuple(item.sample_id for item in batch)
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

    def execute(self, item: WorkItem) -> WorkResult:
        semantic = self.context.semantic_settings
        samples: list[PromptSample] = item.payload
        prompts = [sample.model_prompt for sample in samples]
        stop_strings = semantic["prompt_population"]["generation"].get(
            "stop_strings"
        )
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
            records = self._gsm8k_records(samples, texts)
        else:
            records = self._humaneval_records(samples, texts)
        return WorkResult(records=tuple(records))

    def _gsm8k_records(
        self,
        samples: list[PromptSample],
        texts: list[str],
    ) -> list[JsonObject]:
        records = []
        for sample, text in zip(samples, texts):
            prediction = extract_gsm8k_answer(text)
            ground_truth = extract_gsm8k_answer(sample.payload["answer"])
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "question": sample.payload["question"],
                    "prompt_text": sample.model_prompt,
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
        samples: list[PromptSample],
        texts: list[str],
    ) -> list[JsonObject]:
        completions = [cleanup_code_official(text) for text in texts]
        timeout = self.context.semantic_settings["task_evaluation"][
            "execution_timeout"
        ]

        def evaluate(arguments: tuple[PromptSample, str]) -> JsonObject:
            sample, completion = arguments
            return check_correctness(
                sample.payload["problem"],
                completion,
                timeout,
                completion_id=0,
            )

        workers = self.context.execution_settings["evaluation_workers"]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            evaluations = list(
                executor.map(evaluate, zip(samples, completions))
            )
        records = []
        for sample, text, completion, evaluation in zip(
            samples, texts, completions, evaluations
        ):
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "task_id": sample.payload["task_id"],
                    "prompt_text": sample.model_prompt,
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
                    "num_shots": self._num_shots(),
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
                    "num_shots": self._num_shots(),
                }
            )
        return summary

    def _num_shots(self) -> int:
        return int(
            self.context.semantic_settings["prompt_population"][
                "prompt_policy"
            ]["parameters"]["num_shots"]
        )

    def close(self) -> None:
        pass
