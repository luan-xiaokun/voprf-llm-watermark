# adapted from https://github.com/jthickstun/watermark/blob/main/demo/detect.py

import sys
import time
from dataclasses import dataclass

import numpy as np
import pyximport
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..detector import DetectionCost, DetectionResult, WatermarkDetector

pyximport.install(
    reload_support=True,
    language_level=sys.version_info[0],
    setup_args={"include_dirs": np.get_include()},
)
from .optimized_mersenne import MersenneRNG as mersenne_rng  # type: ignore
from .levenshtein import levenshtein  # type: ignore
from .optimized_levenshtein import detect_cython  # type: ignore


@dataclass
class RDFDetectionResult(DetectionResult):
    milestones: list[int] | None = None
    p_value_at_milestones: list[float] | None = None


@dataclass
class RDFDetectionCost(DetectionCost):
    n_runs: int


def detect(tokens, n, k, xi, gamma=0.0):
    m = len(tokens)
    # ensure k is larger than m
    if k > m:
        return float("inf")

    A = np.empty((m - (k - 1), n))
    for i in range(m - (k - 1)):
        for j in range(n):
            indices = (j + np.arange(k)) % n
            A[i][j] = levenshtein(tokens[i : i + k], xi[indices], gamma)

    return np.min(A)


def permutation_test_incremental(
    tokens: list[int],
    key: int,
    n: int,
    vocab_size: int,
    n_runs: int = 100,
    step_size: int | None = None,
) -> float | dict[str, list]:
    total_tokens = len(tokens)
    print(f"total tokens: {total_tokens}")
    tokens_np = np.array(tokens, dtype=np.int64)
    rng = mersenne_rng(key)
    xi = np.array(
        [rng.rand() for _ in range(n * vocab_size)], dtype=np.float32
    ).reshape(n, vocab_size)

    # original detection logic here
    if step_size is None or step_size <= 0:
        # k = total_tokens
        test_result = detect_cython(tokens_np, n, total_tokens, xi)
        p_val_count = 0
        for _ in range(n_runs):
            xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
            null_result = detect_cython(tokens_np, n, total_tokens, xi_alternative)
            p_val_count += null_result <= test_result
        print(f"p-value: {(p_val_count + 1.0) / (n_runs + 1.0)}")
        return (p_val_count + 1.0) / (n_runs + 1.0)

    # for incremental detection
    milestones = list(range(step_size, total_tokens, step_size))
    print(milestones)
    milestones = sorted(list(set(milestones + [total_tokens])))

    test_stats_at_milestones = np.array(
        [detect_cython(tokens_np[:t], n, k=t, xi=xi) for t in milestones]
    )

    p_val_counts = np.zeros(len(milestones), dtype=int)
    for _ in range(n_runs):
        xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)

        null_stats_at_milestones = np.array(
            [
                detect_cython(tokens_np[:t], n, k=t, xi=xi_alternative)
                for t in milestones
            ]
        )

        p_val_counts += null_stats_at_milestones <= test_stats_at_milestones

    p_values = (p_val_counts + 1.0) / (n_runs + 1.0)
    print(len(p_values), p_values)

    return {"milestones": milestones, "p_values": p_values.tolist()}


def permutation_test(tokens, key, n, k, vocab_size, n_runs=100):
    rng = mersenne_rng(key)
    xi = np.array(
        [rng.rand() for _ in range(n * vocab_size)], dtype=np.float32
    ).reshape(n, vocab_size)
    test_result = detect(tokens, n, k, xi)

    p_val = 0
    for run in range(n_runs):
        xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
        null_result = detect(tokens, n, k, xi_alternative)

        # assuming lower test values indicate presence of watermark
        p_val += null_result <= test_result

    return (p_val + 1.0) / (n_runs + 1.0)


class RDFDetector(WatermarkDetector):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        length: int = 256,
        seed: int = 42,
        n_runs: int = 100,
    ):
        self.tokenizer = tokenizer
        self.length = length
        self.seed = seed
        self.n_runs = n_runs


    def detect(
        self,
        text: str,
        n: int | None = None,
        n_runs: int | None = None,
        step_size: int | None = None,
        token_num: int | None = None,
    ) -> RDFDetectionResult:
        n = n or self.length
        n_runs = n_runs or self.n_runs
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if token_num is None:
            token_num = len(tokens)
        tokens = tokens[:token_num]
        key = self.seed
        vocab_size = len(self.tokenizer.get_vocab())

        result = permutation_test_incremental(
            tokens, key, n, vocab_size, n_runs=n_runs, step_size=step_size
        )

        # without step_size, return single result
        if isinstance(result, float):
            return RDFDetectionResult(total_token_num=len(tokens), p_value=result)

        milestones = result["milestones"]
        p_values = result["p_values"]

        return RDFDetectionResult(
            total_token_num=len(tokens),
            p_value=p_values[-1],
            milestones=milestones,
            p_values=p_values,
        )

    def batch_detect(
        self,
        texts: list[str],
        n: int | None = None,
        n_runs: int | None = None,
        step_size: int | None = None,
        token_num: int | None = None,
    ) -> tuple[list[RDFDetectionResult], list[RDFDetectionCost]]:
        n = n or self.length
        n_runs = n_runs or self.n_runs
        key = self.seed
        vocab_size = len(self.tokenizer.get_vocab())

        detection_results = []
        detection_costs = []

        for text in tqdm(texts):
            start_time = time.perf_counter()

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if token_num is None:
                token_num = len(tokens)
            tokens = tokens[:token_num]
            k = len(tokens)
            result = permutation_test_incremental(
                tokens, key, n, vocab_size, n_runs=n_runs, step_size=step_size
            )

            end_time = time.perf_counter()

            if isinstance(result, float):
                detection_result = RDFDetectionResult(
                    total_token_num=len(tokens), p_value=result
                )
            else:
                detection_result = RDFDetectionResult(
                    total_token_num=len(tokens),
                    p_value=result["p_values"][-1],
                    milestones=result["milestones"],
                    p_value_at_milestones=result["p_values"],
                )
            detection_costs.append(
                RDFDetectionCost(
                    token_num=k, total_time=end_time - start_time, n_runs=n_runs
                )
            )
            detection_results.append(detection_result)

        return detection_results, detection_costs
