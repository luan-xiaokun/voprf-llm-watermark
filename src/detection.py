import math
import struct
from collections.abc import Callable
from typing import NamedTuple

from scipy import stats
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from voprf_py import (
    BlindedElement,
    EvaluationElement,
    Proof,
    VoprfServer,
    PublicKey,
    finalize_batch_blind_results,
    prepare_batch_blind_inputs,
)


class DetectionResult(NamedTuple):
    green_token_num: int
    total_token_num: int
    z_score: float
    p_value: float


def _local_detect(
    token_ids: list[int],
    window_size: int,
    gamma: float,
    voprf_server: VoprfServer,
) -> int:
    msg_inputs = [
        struct.pack(f">{window_size + 1}I", *token_ids[i - window_size: i + 1])
        for i in range(window_size, len(token_ids))
    ]
    msg_hashes = voprf_server.batch_evaluate(msg_inputs)
    probs = [int.from_bytes(h, "big") / (1 << 8 * len(h)) for h in msg_hashes]
    is_green = [p < gamma for p in probs]

    return sum(is_green)


def compute_z_score_and_p_value(
    green_token_num: int,
    total_token_num: int,
    gamma: float,
) -> tuple[float, float]:
    if total_token_num <= 0 or abs(gamma - 0.5) >= 0.5:
        return float("nan"), float("nan")

    expected_green = total_token_num * gamma
    variance = total_token_num * gamma * (1.0 - gamma)

    z_score = (green_token_num - expected_green) / math.sqrt(variance)
    p_value = float(stats.norm.sf(z_score))

    return z_score, p_value


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
) -> list[DetectionResult]:
    voprf_server = VoprfServer(seed)
    token_ids_list = [
        tokenizer.encode(text, add_special_tokens=False) for text in texts
    ]
    results = []
    for token_ids in token_ids_list:
        green_token_num = _local_detect(token_ids, window_size, gamma, voprf_server)
        z_score, p_value = compute_z_score_and_p_value(
            green_token_num, len(token_ids) - window_size, gamma
        )
        results.append(
            DetectionResult(
                green_token_num=green_token_num,
                total_token_num=len(token_ids) - window_size,
                z_score=z_score,
                p_value=p_value,
            )
        )

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
) -> list[DetectionResult]:
    token_ids_list = [
        tokenizer.encode(text, add_special_tokens=False) for text in texts
    ]
    msg_inputs_list = [
        [
            struct.pack(f">{window_size + 1}I", *token_ids[i - window_size - 1 : i])
            for i in range(window_size + 1, len(token_ids))
        ]
        for token_ids in token_ids_list
    ]
    results = []
    for msg_inputs in msg_inputs_list:
        states, blinded_elements = prepare_batch_blind_inputs(msg_inputs)
        messages, proof = server_interface(blinded_elements)
        msg_hashes = finalize_batch_blind_results(
            msg_inputs, states, messages, proof, server_public_key
        )
        probs = [int.from_bytes(h, "big") / (1 << 8 * len(h)) for h in msg_hashes]
        is_green = [p < gamma for p in probs]
        green_token_num = sum(is_green)
        z_score, p_value = compute_z_score_and_p_value(
            green_token_num, len(msg_inputs), gamma
        )
        results.append(
            DetectionResult(
                green_token_num=green_token_num,
                total_token_num=len(msg_inputs),
                z_score=z_score,
                p_value=p_value,
            )
        )
    return results



