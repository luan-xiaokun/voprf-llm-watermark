import itertools
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass

from scipy import special
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from tqdm import tqdm
from voprf_py import (
    BlindedElement,
    EvaluationElement,
    Proof,
    PublicKey,
    VoprfServer,
    finalize_batch_blind_results,
    prepare_batch_blind_inputs,
)

from ..detector import DetectionCost, DetectionError, DetectionResult, WatermarkDetector

HASH_BITS = 512


class HashToCurveFailure(DetectionError):
    pass


class BlindEvaluationError(DetectionError):
    pass


class InValidProof(DetectionError):
    pass


@dataclass
class VOWDetectionResult(DetectionResult):
    green_token_num: int
    effective_token_num: int
    green_ratio: float
    p_values_per_token: list[float] | None
    green_token_mask: list[bool] | None


@dataclass
class VOWDetectionCost(DetectionCost):
    raw_string_bytes: int
    blinded_elements_bytes: int
    evaluation_elements_bytes: int
    proof_bytes: int
    total_communication_bytes: int
    preparation_time: float
    evaluation_time: float
    finalization_time: float


class VOWDetector(WatermarkDetector):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seed: bytes,
        gamma: float,
        window_size: int = 7,
    ):
        self.tokenizer = tokenizer
        self.seed = int.from_bytes(seed, "big")
        self.gamma = gamma
        self.window_size = window_size
        self.voprf_server = VoprfServer(seed)

    def local_detect(
        self,
        text: str,
        gamma: float | None = None,
        window_size: int | None = None,
        include_p_values_per_token: bool = False,
        return_green_token_mask: bool = False,
        token_num: int | None = None,
        step_size: int | None = None,
    ) -> VOWDetectionResult:
        return self.local_batch_detect(
            [text],
            gamma=gamma,
            window_size=window_size,
            include_p_values_per_token=include_p_values_per_token,
            return_green_token_mask=return_green_token_mask,
            token_num=token_num,
            step_size=step_size,
        )[0]

    def local_batch_detect(
        self,
        texts: list[str],
        gamma: float | None = None,
        window_size: int | None = None,
        include_p_values_per_token: bool = False,
        return_green_token_mask: bool = False,
        token_num: int | None = None,
        step_size: int | None = None,
    ) -> list[VOWDetectionResult]:
        gamma = gamma or self.gamma
        window_size = window_size or self.window_size

        if step_size is not None and step_size > 0:
            include_p_values_per_token = True

        token_ids_list = [
            self.tokenizer.encode(text, add_special_tokens=False) for text in texts
        ]
        if token_num is not None:
            token_ids_list = [ids[:token_num] for ids in token_ids_list]

        n_grams_list = [
            [
                tuple(token_ids[i - window_size : i + 1])
                for i in range(window_size, len(token_ids))
            ]
            for token_ids in token_ids_list
        ]
        unique_n_grams_list = [list(dict.fromkeys(n_grams)) for n_grams in n_grams_list]
        indices = list(
            itertools.accumulate(
                (len(n_grams) for n_grams in unique_n_grams_list), initial=0
            )
        )
        flatten_msg_inputs = [
            struct.pack(f">{window_size + 1}I", *n_gram)
            for n_grams in unique_n_grams_list
            for n_gram in n_grams
        ]
        flatten_msg_hashes = self.voprf_server.batch_evaluate(flatten_msg_inputs)
        msg_hashes_list = [
            flatten_msg_hashes[indices[i] : indices[i + 1]]
            for i in range(len(indices) - 1)
        ]

        threshold = gamma * (1 << HASH_BITS)
        threshold_bytes = int(threshold).to_bytes(HASH_BITS // 8, "big")

        results = []
        for i, msg_hashes in enumerate(msg_hashes_list):
            gram_to_color_map = {
                gram: h < threshold_bytes
                for gram, h in zip(unique_n_grams_list[i], msg_hashes)
            }

            if not include_p_values_per_token:
                effective_token_num = len(unique_n_grams_list[i])
                green_token_num = sum(gram_to_color_map.values())

                p_values = None
                p_value = float(
                    special.betainc(
                        green_token_num,
                        effective_token_num - green_token_num + 1,
                        gamma,
                    )
                )
            else:
                green_token_num = 0
                effective_token_num = 0
                p_values = []
                seen_grams = set()

                for gram in n_grams_list[i]:
                    if gram not in seen_grams:
                        seen_grams.add(gram)
                        green_token_num += gram_to_color_map[gram]
                        effective_token_num += 1

                        p_value = float(
                            special.betainc(
                                green_token_num,
                                effective_token_num - green_token_num + 1,
                                gamma,
                            )
                        )
                    else:
                        p_value = p_values[-1] if p_values else 1.0
                    p_values.append(p_value)
                p_value = p_values[-1] if p_values else 1.0

            green_token_mask = None
            if return_green_token_mask:
                green_token_mask = [False] * window_size
                for j in range(window_size, len(token_ids_list[i])):
                    n_gram = tuple(token_ids_list[i][j - window_size : j + 1])
                    color = gram_to_color_map[n_gram]
                    green_token_mask.append(color)

            if effective_token_num > 0:
                green_ratio = green_token_num / effective_token_num
            else:
                green_ratio = 0.0
            detect_result = VOWDetectionResult(
                green_token_num=green_token_num,
                effective_token_num=effective_token_num,
                total_token_num=len(token_ids_list[i]),
                green_ratio=green_ratio,
                p_value=p_value,
                p_values_per_token=p_values,
                green_token_mask=green_token_mask,
            )
            if step_size is not None and step_size > 0 and p_values:
                milestones = list(
                    range(step_size, len(token_ids_list[i]) + 1, step_size)
                )
                valid_milestones = [m for m in milestones if m > window_size]
                indices = [m - window_size - 1 for m in valid_milestones]
                indices = [idx for idx in indices if idx < len(p_values)]
                final_milestones = [valid_milestones[k] for k in range(len(indices))]

                detect_result.step_size = step_size
                detect_result.milestones = final_milestones
                detect_result.step_p_values = [p_values[idx] for idx in indices]

            results.append(detect_result)

        return results

    def detect(
        self,
        text: str,
        server_public_key: PublicKey,
        server_interface: Callable[
            [list[BlindedElement]], tuple[list[EvaluationElement], Proof]
        ],
        gamma: float | None = None,
        window_size: int | None = None,
        token_num: int | None = None,
        step_size: int | None = None,
    ) -> VOWDetectionResult:
        results, _ = self.batch_detect(
            [text],
            server_public_key=server_public_key,
            server_interface=server_interface,
            gamma=gamma,
            window_size=window_size,
            token_num=token_num,
            step_size=step_size,
        )
        return results[0]

    def batch_detect(
        self,
        texts: list[str],
        server_public_key: PublicKey,
        server_interface: Callable[
            [list[BlindedElement]], tuple[list[EvaluationElement], Proof]
        ],
        gamma: float | None = None,
        window_size: int | None = None,
        token_num: int | None = None,
        step_size: int | None = None,
    ) -> tuple[list[VOWDetectionResult], list[VOWDetectionCost]]:
        gamma = gamma or self.gamma
        window_size = window_size or self.window_size

        threshold = gamma * (1 << HASH_BITS)
        threshold_bytes = int(threshold).to_bytes(HASH_BITS // 8, "big")
        detection_results = []
        detection_costs = []

        for text in tqdm(texts, desc="Detecting"):
            preparation_start = time.perf_counter()
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            current_token_num = len(token_ids) if token_num is None else token_num
            token_ids = token_ids[:current_token_num]
            n_grams = [
                tuple(token_ids[i - window_size : i + 1])
                for i in range(window_size, len(token_ids))
            ]
            # this works for Python 3.7+
            unique_n_grams = list(dict.fromkeys(n_grams))

            msg_inputs = [
                struct.pack(f">{window_size + 1}I", *n_gram)
                for n_gram in unique_n_grams
            ]

            try:
                states, blinded_elements = prepare_batch_blind_inputs(msg_inputs)
                blinded_elements_bytes = sum(
                    len(be.to_binary()) for be in blinded_elements
                )
            except ValueError:
                raise HashToCurveFailure(
                    "Failed to hash to curve, please check the input."
                )

            preparation_end = time.perf_counter()
            preparation_time = preparation_end - preparation_start
            evaluation_start = time.perf_counter()

            try:
                messages, proof = server_interface(blinded_elements)
                evaluation_elements_bytes = sum(len(ee.to_binary()) for ee in messages)
                proof_bytes = len(proof.to_binary())
            except ValueError:
                raise BlindEvaluationError(
                    "Failed to evaluate blinded elements, check the server interface."
                )

            evaluation_end = time.perf_counter()
            evaluation_time = evaluation_end - evaluation_start
            total_communication_bytes = (
                blinded_elements_bytes + evaluation_elements_bytes + proof_bytes
            )
            finalization_start = time.perf_counter()

            try:
                msg_hashes = finalize_batch_blind_results(
                    msg_inputs, states, messages, proof, server_public_key
                )
            except ValueError:
                raise InValidProof(
                    "The proof is invalid, please check the server interface."
                )

            green_token_num = sum(h < threshold_bytes for h in msg_hashes)

            p_values = None
            if step_size is not None and step_size > 0:
                gram_to_color_map = {
                    gram: h < threshold_bytes
                    for gram, h in zip(unique_n_grams, msg_hashes)
                }
                p_values = []
                cur_green = 0
                cur_effective = 0
                seen_grams = set()
                for gram in n_grams:
                    if gram not in seen_grams:
                        seen_grams.add(gram)
                        cur_green += gram_to_color_map[gram]
                        cur_effective += 1
                        val = float(
                            special.betainc(
                                cur_green, cur_effective - cur_green + 1, gamma
                            )
                        )
                    else:
                        val = p_values[-1] if p_values else 1.0
                    p_values.append(val)

            effective_token_num = len(unique_n_grams)
            total_token_num = len(token_ids)
            if effective_token_num > 0:
                green_ratio = green_token_num / effective_token_num
            else:
                green_ratio = 0.0

            p_value = float(
                special.betainc(
                    green_token_num, effective_token_num - green_token_num + 1, gamma
                )
            )

            finalization_end = time.perf_counter()
            finalization_time = finalization_end - finalization_start
            total_time = finalization_end - preparation_start

            detect_result = VOWDetectionResult(
                green_token_num=green_token_num,
                effective_token_num=effective_token_num,
                total_token_num=total_token_num,
                green_ratio=green_ratio,
                p_value=p_value,
                p_values_per_token=p_values,
                green_token_mask=None,
            )
            if step_size is not None and step_size > 0 and p_values:
                milestones = list(range(step_size, total_token_num + 1, step_size))
                valid_milestones = [m for m in milestones if m > window_size]
                indices = [m - window_size - 1 for m in valid_milestones]
                indices = [idx for idx in indices if idx < len(p_values)]
                final_milestones = [valid_milestones[k] for k in range(len(indices))]
                detect_result.step_size = step_size
                detect_result.milestones = final_milestones
                detect_result.step_p_values = [p_values[idx] for idx in indices]

            detect_cost = VOWDetectionCost(
                raw_string_bytes=len(text.encode("utf-8")),
                token_num=len(token_ids),
                blinded_elements_bytes=blinded_elements_bytes,
                evaluation_elements_bytes=evaluation_elements_bytes,
                proof_bytes=proof_bytes,
                total_communication_bytes=total_communication_bytes,
                preparation_time=preparation_time,
                evaluation_time=evaluation_time,
                finalization_time=finalization_time,
                total_time=total_time,
            )
            detection_results.append(detect_result)
            detection_costs.append(detect_cost)

        return detection_results, detection_costs
