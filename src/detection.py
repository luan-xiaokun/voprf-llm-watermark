import itertools
import struct
import time
from collections.abc import Callable
from typing import NamedTuple

from scipy import special
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from voprf_py import (
    BlindedElement,
    EvaluationElement,
    Proof,
    PublicKey,
    VoprfServer,
    finalize_batch_blind_results,
    prepare_batch_blind_inputs,
)

HASH_BITS = 512


class DetectionError(Exception):
    pass


class HashToCurveFailure(DetectionError):
    pass


class BlindEvaluationError(DetectionError):
    pass


class InValidProof(DetectionError):
    pass


class DetectionResult(NamedTuple):
    green_token_num: int
    effective_token_num: int
    total_token_num: int
    green_ratio: float
    p_value: float
    p_values_per_token: list[float] | None


class DetectionCost(NamedTuple):
    raw_string_bytes: int
    token_ids_length: int
    blinded_elements_bytes: int
    evaluation_elements_bytes: int
    proof_bytes: int
    total_communication_bytes: int
    preparation_time: float
    evaluation_time: float
    finalization_time: float
    total_time: float


def local_detect_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    window_size: int,
    gamma: float,
    seed: bytes,
) -> DetectionResult:
    return local_detect_batch_text([text], tokenizer, window_size, gamma, seed)[0]


def local_detect_batch_text(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    window_size: int,
    gamma: float,
    seed: bytes,
    include_p_values_per_token: bool = False,
) -> list[DetectionResult]:
    results = []
    voprf_server = VoprfServer(seed)
    token_ids_list = [
        tokenizer.encode(text, add_special_tokens=False) for text in texts
    ]

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
        for n_grams in n_grams_list
        for n_gram in n_grams
    ]
    flatten_msg_hashes = voprf_server.batch_evaluate(flatten_msg_inputs)
    msg_hashes_list = [
        flatten_msg_hashes[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)
    ]

    threshold = gamma * (1 << HASH_BITS)
    threshold_bytes = int(threshold).to_bytes(HASH_BITS // 8, "big")

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
                    green_token_num, effective_token_num - green_token_num + 1, gamma
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

        if effective_token_num > 0:
            green_ratio = green_token_num / effective_token_num
        else:
            green_ratio = 0.0

        detect_result = DetectionResult(
            green_token_num=green_token_num,
            effective_token_num=effective_token_num,
            total_token_num=len(token_ids_list[i]),
            green_ratio=green_ratio,
            p_value=p_value,
            p_values_per_token=p_values,
        )
        results.append(detect_result)

    return results


def client_detect_batch_text(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    window_size: int,
    gamma: float,
    server_public_key: PublicKey,
    server_interface: Callable[
        [list[BlindedElement]], tuple[list[EvaluationElement], Proof]
    ],
) -> tuple[list[DetectionResult], list[DetectionCost]]:
    threshold = gamma * (1 << HASH_BITS)
    threshold_bytes = int(threshold).to_bytes(HASH_BITS // 8, "big")
    detection_results = []
    detection_costs = []

    for text in texts:
        preparation_start = time.perf_counter()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        n_grams = [
            tuple(token_ids[i - window_size : i + 1])
            for i in range(window_size, len(token_ids))
        ]
        # this works for Python 3.7+
        unique_n_grams = list(dict.fromkeys(n_grams))

        msg_inputs = [
            struct.pack(f">{window_size + 1}I", *n_gram) for n_gram in unique_n_grams
        ]

        try:
            states, blinded_elements = prepare_batch_blind_inputs(msg_inputs)
            blinded_elements_bytes = sum(len(be.to_binary()) for be in blinded_elements)
        except ValueError:
            raise HashToCurveFailure("Failed to hash to curve, please check the input.")

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

        detect_result = DetectionResult(
            green_token_num=green_token_num,
            effective_token_num=effective_token_num,
            total_token_num=total_token_num,
            green_ratio=green_ratio,
            p_value=p_value,
            p_values_per_token=None,
        )
        detect_cost = DetectionCost(
            raw_string_bytes=len(text.encode("utf-8")),
            token_ids_length=len(token_ids),
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
