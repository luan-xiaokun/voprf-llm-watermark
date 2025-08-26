# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain, tee
from math import sqrt

import scipy.stats
import torch
from tokenizers import Tokenizer

from ..detector import DetectionCost, DetectionResult, WatermarkDetector
from .base import WatermarkBase
from .normalizers import normalization_strategy_lookup


@dataclass
class KGWDetectionResult(DetectionResult):
    green_token_num: int
    effective_token_num: int
    green_ratio: float
    z_score: float
    green_token_mask: list[bool] | None = None
    window_list: list[float] | None = None
    z_score_at_T: torch.Tensor | None = None


class KGWDetectionCost(DetectionCost):
    pass


class KGWDetector(WatermarkBase, WatermarkDetector):
    """This is the detector for all watermarks imprinted with WatermarkLogitsProcessor.

    The detector needs to be given the exact same settings that were given during text generation  to replicate the watermark
    greenlist generation and so detect the watermark.
    This includes the correct device that was used during text generation, the correct tokenizer, the correct
    seeding_scheme name, and parameters (delta, gamma).

    Optional arguments are
    * normalizers ["unicode", "homoglyphs", "truecase"] -> These can mitigate modifications to generated text that could trip the watermark
    * ignore_repeated_ngrams -> This option changes the detection rules to count every unique ngram only once.
    * z_threshold -> Changing this threshold will change the sensitivity of the detector.
    """

    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = [
            "unicode"
        ],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_ngrams: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert (
            tokenizer
        ), "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)
        self.seed = self.hash_key

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(
                normalization_strategy_lookup(normalization_strategy)
            )
        self.ignore_repeated_ngrams = ignore_repeated_ngrams

    def dummy_detect(
        self,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_all_window_scores: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
    ):
        # HF-style output dictionary
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=float("nan")))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=float("nan")))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=float("nan")))
        if return_z_score:
            score_dict.update(dict(z_score=float("nan")))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = float("nan")
            score_dict.update(dict(p_value=float("nan")))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=[]))
        if return_all_window_scores:
            score_dict.update(dict(window_list=[]))
        if return_z_at_T:
            score_dict.update(dict(z_score_at_T=torch.tensor([])))

        output_dict = {}
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert (
                z_threshold is not None
            ), "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = False

        return output_dict

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, prefix: tuple[int], target: int):
        """Expensive re-seeding and sampling is cached."""
        # Handle with care, should ideally reset on __getattribute__ access to self.prf_type, self.context_width, self.self_salt, self.hash_key
        greenlist_ids = self._get_greenlist_ids(
            torch.as_tensor(prefix, device=self.device)
        )
        return True if target in greenlist_ids else False

    def _score_ngrams_in_passage(self, input_ids: torch.Tensor):
        """Core function to gather all ngrams in the input and compute their watermark."""
        if len(input_ids) - self.context_width < 1:
            print(input_ids)
            print(self.tokenizer.decode(input_ids))
            raise ValueError(
                f"Must have at least {1} token to score after "
                f"the first min_prefix_len={self.context_width} tokens required by the seeding scheme."
            )

        # Compute scores for all ngrams contexts in the passage:
        token_ngram_generator = ngrams(
            input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt
        )
        frequencies_table = collections.Counter(token_ngram_generator)
        ngram_to_watermark_lookup = {}
        for idx, ngram_example in enumerate(frequencies_table.keys()):
            prefix = ngram_example if self.self_salt else ngram_example[:-1]
            target = ngram_example[-1]
            ngram_to_watermark_lookup[ngram_example] = self._get_ngram_score_cached(
                prefix, target
            )

        return ngram_to_watermark_lookup, frequencies_table

    def _get_green_at_T_booleans(
        self, input_ids, ngram_to_watermark_lookup
    ) -> tuple[torch.Tensor]:
        """Generate binary list of green vs. red per token, a separate list that ignores repeated ngrams, and a list of offsets to
        convert between both representations:
        green_token_mask = green_token_mask_unique[offsets] except for all locations where otherwise a repeat would be counted
        """
        green_token_mask, green_token_mask_unique, offsets = [], [], []
        used_ngrams = {}
        unique_ngram_idx = 0
        ngram_examples = ngrams(
            input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt
        )

        for idx, ngram_example in enumerate(ngram_examples):
            green_token_mask.append(ngram_to_watermark_lookup[ngram_example])
            if self.ignore_repeated_ngrams:
                if ngram_example in used_ngrams:
                    pass
                else:
                    used_ngrams[ngram_example] = True
                    unique_ngram_idx += 1
                    green_token_mask_unique.append(
                        ngram_to_watermark_lookup[ngram_example]
                    )
            else:
                green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
                unique_ngram_idx += 1
            offsets.append(unique_ngram_idx - 1)
        return (
            torch.tensor(green_token_mask),
            torch.tensor(green_token_mask_unique),
            torch.tensor(offsets),
        )

    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
    ):
        ngram_to_watermark_lookup, frequencies_table = self._score_ngrams_in_passage(
            input_ids
        )
        green_token_mask, green_unique, offsets = self._get_green_at_T_booleans(
            input_ids, ngram_to_watermark_lookup
        )

        # Count up scores over all ngrams
        if self.ignore_repeated_ngrams:
            # Method that only counts a green/red hit once per unique ngram.
            # New num total tokens scored (T) becomes the number unique ngrams.
            # We iterate over all unqiue token ngrams in the input, computing the greenlist
            # induced by the context in each, and then checking whether the last
            # token falls in that greenlist.
            num_tokens_scored = len(frequencies_table.keys())
            green_token_count = sum(ngram_to_watermark_lookup.values())
        else:
            num_tokens_scored = sum(frequencies_table.values())
            assert (
                num_tokens_scored
                == len(input_ids) - self.context_width + self.self_salt
            )
            green_token_count = sum(
                freq * outcome
                for freq, outcome in zip(
                    frequencies_table.values(), ngram_to_watermark_lookup.values()
                )
            )
        assert green_token_count == green_unique.sum()

        # HF-style output dictionary
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(
                dict(green_fraction=(green_token_count / num_tokens_scored))
            )
        if return_z_score:
            score_dict.update(
                dict(
                    z_score=self._compute_z_score(green_token_count, num_tokens_scored)
                )
            )
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask.tolist()))
        if return_z_at_T:
            # Score z_at_T separately:
            sizes = torch.arange(1, len(green_unique) + 1)
            seq_z_score_enum = torch.cumsum(green_unique, dim=0) - self.gamma * sizes
            seq_z_score_denom = torch.sqrt(sizes * self.gamma * (1 - self.gamma))
            z_score_at_effective_T = seq_z_score_enum / seq_z_score_denom
            z_score_at_T = z_score_at_effective_T[offsets]
            assert torch.isclose(z_score_at_T[-1], torch.tensor(z_score))

            score_dict.update(dict(z_score_at_T=z_score_at_T))

        return score_dict

    def _score_windows_impl_batched(
        self,
        input_ids: torch.Tensor,
        window_size: str,
        window_stride: int = 1,
    ):
        # Implementation details:
        # 1) --ignore_repeated_ngrams is applied globally, and windowing is then applied over the reduced binary vector
        #      this is only one way of doing it, another would be to ignore bigrams within each window (maybe harder to parallelize that)
        # 2) These windows on the binary vector of green/red hits, independent of context_width, in contrast to Kezhi's first implementation
        # 3) z-scores from this implementation cannot be directly converted to p-values, and should only be used as labels for a
        #    ROC chart that calibrates to a chosen FPR. Due, to windowing, the multiple hypotheses will increase scores across the board#
        #    naive_count_correction=True is a partial remedy to this

        ngram_to_watermark_lookup, frequencies_table = self._score_ngrams_in_passage(
            input_ids
        )
        green_mask, green_ids, offsets = self._get_green_at_T_booleans(
            input_ids, ngram_to_watermark_lookup
        )
        len_full_context = len(green_ids)

        partial_sum_id_table = torch.cumsum(green_ids, dim=0)

        if window_size == "max":
            # could start later, small window sizes cannot generate enough power
            # more principled: solve (T * Spike_Entropy - g * T) / sqrt(T * g * (1 - g)) = z_thresh for T
            sizes = range(1, len_full_context)
        else:
            sizes = [int(x) for x in window_size.split(",") if len(x) > 0]

        z_score_max_per_window = torch.zeros(len(sizes))
        cumulative_eff_z_score = torch.zeros(len_full_context)
        s = window_stride

        window_fits = False
        for idx, size in enumerate(sizes):
            if size <= len_full_context:
                # Compute hits within window for all positions in parallel:
                window_score = torch.zeros(
                    len_full_context - size + 1, dtype=torch.long
                )
                # Include 0-th window
                window_score[0] = partial_sum_id_table[size - 1]
                # All other windows from the 1st:
                window_score[1:] = (
                    partial_sum_id_table[size::s] - partial_sum_id_table[:-size:s]
                )

                # Now compute batched z_scores
                batched_z_score_enum = window_score - self.gamma * size
                z_score_denom = sqrt(size * self.gamma * (1 - self.gamma))
                batched_z_score = batched_z_score_enum / z_score_denom

                # And find the maximal hit
                maximal_z_score = batched_z_score.max()
                z_score_max_per_window[idx] = maximal_z_score

                z_score_at_effective_T = torch.cummax(batched_z_score, dim=0)[0]
                cumulative_eff_z_score[size::s] = torch.maximum(
                    cumulative_eff_z_score[size::s], z_score_at_effective_T[:-1]
                )
                window_fits = True  # successful computation for any window in sizes

        if not window_fits:
            raise ValueError(
                f"Could not find a fitting window with window sizes {window_size} for (effective) context length {len_full_context}."
            )

        # Compute optimal window size and z-score
        cumulative_z_score = cumulative_eff_z_score[offsets]
        optimal_z, optimal_window_size_idx = z_score_max_per_window.max(dim=0)
        optimal_window_size = sizes[optimal_window_size_idx]
        return (
            optimal_z,
            optimal_window_size,
            z_score_max_per_window,
            cumulative_z_score,
            green_mask,
        )

    def _score_sequence_window(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
        window_size: str = None,
        window_stride: int = 1,
    ):
        (
            optimal_z,
            optimal_window_size,
            _,
            z_score_at_T,
            green_mask,
        ) = self._score_windows_impl_batched(input_ids, window_size, window_stride)

        # HF-style output dictionary
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=optimal_window_size))

        denom = sqrt(optimal_window_size * self.gamma * (1 - self.gamma))
        green_token_count = int(optimal_z * denom + self.gamma * optimal_window_size)
        green_fraction = green_token_count / optimal_window_size
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=green_fraction))
        if return_z_score:
            score_dict.update(dict(z_score=optimal_z))
        if return_z_at_T:
            score_dict.update(dict(z_score_at_T=z_score_at_T))
        if return_p_value:
            z_score = score_dict.get("z_score", optimal_z)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))

        # Return per-token results for mask. This is still the same, just scored by windows
        # todo would be to mark the actually counted tokens differently
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_mask.tolist()))

        return score_dict

    @staticmethod
    def _build_detection_result_from_dict(output_dict: dict):
        p_value = output_dict.get("p_value", float("nan"))
        total_token_num = output_dict.get("total_token_num", 0)
        green_token_num = output_dict.get("num_green_tokens", 0)
        effective_token_num = output_dict.get("num_tokens_scored", 0)
        green_ratio = output_dict.get("green_fraction", 0.0)
        z_score = output_dict.get("z_score", float("nan"))
        green_token_mask = output_dict.get("green_token_mask", None)
        window_list = output_dict.get("window_list", None)
        z_score_at_T = output_dict.get("z_score_at_T", None)

        return KGWDetectionResult(
            p_value=p_value,
            total_token_num=total_token_num,
            green_token_num=green_token_num,
            effective_token_num=effective_token_num,
            green_ratio=green_ratio,
            z_score=z_score,
            green_token_mask=green_token_mask,
            window_list=window_list,
            z_score_at_T=z_score_at_T,
        )

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        token_num: int | None = None,
        window_size: str = None,
        window_stride: int = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        convert_to_float: bool = False,
        return_z_at_T: bool = False,
        **kwargs,
    ) -> KGWDetectionResult:
        """Scores a given string of text and returns a dictionary of results."""

        assert (text is not None) ^ (
            tokenized_text is not None
        ), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = (
                True  # to return the "confidence":=1-p of positive detections
            )

        # run optional normalizers on text
        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            # try to remove the bos_tok at beginning if it's there
            if (self.tokenizer is not None) and (
                tokenized_text[0] == self.tokenizer.bos_token_id
            ):
                tokenized_text = tokenized_text[1:]

        if token_num is None:
            token_num = len(tokenized_text)
        tokenized_text = tokenized_text[:token_num]

        # call score method
        output_dict = {}

        if window_size is not None:
            # assert window_size <= len(tokenized_text) cannot assert for all new types
            score_dict = self._score_sequence_window(
                tokenized_text,
                window_size=window_size,
                window_stride=window_stride,
                return_z_at_T=return_z_at_T,
                **kwargs,
            )
            output_dict.update(score_dict)
        else:
            score_dict = self._score_sequence(
                tokenized_text, return_z_at_T=return_z_at_T, **kwargs
            )
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert (
                z_threshold is not None
            ), "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        # convert any numerical values to float if requested
        if convert_to_float:
            for key, value in output_dict.items():
                if isinstance(value, int):
                    output_dict[key] = float(value)

        total_token_num = len(tokenized_text)
        output_dict.update(dict(total_token_num=total_token_num))

        return self._build_detection_result_from_dict(output_dict)

    def batch_detect(
        self,
        texts: list[str],
        token_num: int | None = None,
        window_size: str = None,
        window_stride: int = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        convert_to_float: bool = False,
        **kwargs,
    ) -> tuple[list[KGWDetectionResult], list[KGWDetectionCost]]:
        results = []
        costs = []

        for text in texts:
            start_time = time.perf_counter()

            result = self.detect(
                text,
                tokenized_text=None,
                token_num=token_num,
                window_size=window_size,
                window_stride=window_stride,
                return_prediction=return_prediction,
                return_scores=return_scores,
                z_threshold=z_threshold,
                convert_to_float=convert_to_float,
                **kwargs,
            )

            end_time = time.perf_counter()
            results.append(result)
            costs.append(
                KGWDetectionCost(
                    token_num=result.total_token_num, total_time=end_time - start_time
                )
            )

        return results, costs


##########################################################################
# Ngram iteration from nltk, extracted to remove the dependency
# Natural Language Toolkit: Utility functions
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Eric Kafe <kafe.eric@gmail.com> (acyclic closures)
# URL: <https://www.nltk.org/>
# For license information, see https://github.com/nltk/nltk/blob/develop/LICENSE.txt
##########################################################################


def ngrams(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n - 1))
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.
