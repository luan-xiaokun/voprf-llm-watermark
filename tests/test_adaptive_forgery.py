import struct
import unittest
from types import SimpleNamespace

import torch
from datasets import Dataset
from voprf_py import VoprfServer

from watermark_suite.attacks import (
    AdaptiveCandidateSelector,
    AdaptiveWatermarkForger,
    VOPRFColorOracle,
    theoretical_green_probability,
    theoretical_queries_per_scored_token,
)
from watermark_suite.core.metrics import calculate_perplexities
from watermark_suite.schemes.vow import VOWDetector
from watermark_suite.experiments.stages.adaptive_forgery import (
    build_sample_metrics,
    build_summary,
)
from watermark_suite.experiments.stages.detection import (
    adaptive_forgery_curve,
)


class MappingOracle:
    def __init__(self, colors: dict[tuple[tuple[int, ...], int], bool]):
        self.colors = colors
        self.calls = []

    def query(self, context: tuple[int, ...], token_id: int) -> bool:
        pair = (context, token_id)
        self.calls.append(pair)
        return self.colors.get(pair, False)


class TinyTokenizer:
    eos_token_id = None
    bos_token_id = 0
    pad_token_id = 0
    all_special_ids = []

    def __call__(self, prompt: str, return_tensors: str):
        del prompt, return_tensors
        return {
            "input_ids": torch.tensor([[5]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return ",".join(str(token_id) for token_id in token_ids)

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [int(piece) for piece in text.split(",")] if text else []


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rankings = [
            [1, 2, 3],
            [2, 3, 4],
            [4, 2, 1],
            [1, 2, 3],
        ]

    @property
    def device(self):
        return torch.device("cpu")

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        del attention_mask, use_cache, return_dict
        step = 0 if past_key_values is None else past_key_values
        ranking = self.rankings[step]
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], 6), -100.0
        )
        for score, token_id in zip([3.0, 2.0, 1.0], ranking):
            logits[:, -1, token_id] = score
        return SimpleNamespace(logits=logits, past_key_values=step + 1)


class UniformCausalModel(torch.nn.Module):
    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        del attention_mask, return_dict
        return SimpleNamespace(
            logits=torch.zeros(
                input_ids.shape[0], input_ids.shape[1], 6
            )
        )


class AdaptiveCandidateSelectorTests(unittest.TestCase):
    def test_first_green_fallback_and_pair_cache(self):
        colors = {
            ((9, 8), 1): False,
            ((9, 8), 2): True,
        }
        oracle = MappingOracle(colors)
        selector = AdaptiveCandidateSelector(
            oracle=oracle, window_size=2, max_candidates=3
        )

        warmup = selector.select(0, [], [3, 2, 1])
        self.assertEqual(warmup.selected_token_id, 3)
        self.assertIsNone(warmup.selected_green)
        self.assertEqual(warmup.oracle_query_count, 0)

        green = selector.select(2, [9, 8], [1, 2, 3])
        self.assertEqual(green.selected_token_id, 2)
        self.assertEqual(green.selected_rank, 2)
        self.assertTrue(green.selected_green)
        self.assertEqual(green.oracle_query_count, 2)

        fallback = selector.select(3, [8, 7], [1, 2, 3])
        self.assertEqual(fallback.selected_token_id, 1)
        self.assertFalse(fallback.selected_green)
        self.assertTrue(fallback.fallback)
        self.assertEqual(fallback.oracle_query_count, 3)

        repeated = selector.select(4, [8, 7], [1, 2, 3])
        self.assertTrue(repeated.fallback)
        self.assertEqual(repeated.oracle_query_count, 0)
        self.assertEqual(repeated.cache_hit_count, 3)
        self.assertEqual(len(oracle.calls), 5)


class AdaptiveWatermarkForgerTests(unittest.TestCase):
    def test_left_to_right_generation_and_accounting(self):
        oracle = MappingOracle(
            {
                ((1,), 2): False,
                ((1,), 3): True,
                ((4,), 1): True,
            }
        )
        forger = AdaptiveWatermarkForger(
            model=TinyCausalModel(),
            tokenizer=TinyTokenizer(),
            oracle=oracle,
            window_size=1,
            max_candidates=3,
        )

        result = forger.forge("prompt", max_new_tokens=4)

        self.assertEqual(result.token_ids, [1, 3, 4, 1])
        self.assertEqual(result.scored_token_num, 3)
        self.assertEqual(result.selected_green_token_num, 2)
        self.assertAlmostEqual(result.selected_green_ratio, 2 / 3)
        self.assertEqual(result.fallback_count, 1)
        self.assertEqual(result.oracle_query_count, 6)
        self.assertEqual(result.color_check_count, 6)
        self.assertEqual(result.generated_token_num, 4)
        self.assertAlmostEqual(result.queries_per_generated_token, 1.5)
        self.assertAlmostEqual(result.queries_per_scored_token, 2.0)
        self.assertAlmostEqual(result.fallback_ratio, 1 / 3)
        self.assertAlmostEqual(result.mean_log_probability_gap, 0.25)
        self.assertGreater(result.local_model_perplexity, 1.0)
        self.assertIsNone(result.oracle_protocol_stats)
        self.assertTrue(result.tokenization_preserved)

        compact = result.to_dict(trace_level="compact")
        self.assertEqual(len(compact["trace"]), 4)
        self.assertNotIn("steps", compact)
        self.assertEqual(
            compact["trace"][-1]["cumulative_oracle_query_count"], 6
        )


def test_adaptive_forgery_curve_combines_cost_pvalue_and_asr():
    records = [
        {
            "adaptive_forgery": {
                "trace": [
                    {
                        "position": index,
                        "cumulative_oracle_query_count": queries,
                        "selected_green": green,
                    }
                    for index, (queries, green) in enumerate(
                        [(0, None), (0, None), (1, True), (3, True)]
                    )
                ]
            },
            "detection": {
                "milestones": [2, 4],
                "step_p_values": [0.1, 0.000001],
            },
        },
        {
            "adaptive_forgery": {
                "trace": [
                    {
                        "position": index,
                        "cumulative_oracle_query_count": queries,
                        "selected_green": green,
                    }
                    for index, (queries, green) in enumerate(
                        [(0, None), (0, None), (2, False), (4, True)]
                    )
                ]
            },
            "detection": {
                "milestones": [2, 4],
                "step_p_values": [0.2, 0.01],
            },
        },
    ]

    curve = adaptive_forgery_curve(records, [0.00001])

    assert curve == [
        {
            "token_num": 2,
            "eligible_sample_num": 2,
            "mean_oracle_query_count": 0,
            "median_oracle_query_count": 0.0,
            "mean_queries_per_token": 0.0,
            "mean_selected_green_token_count": 0,
            "mean_selected_green_ratio": 0.0,
            "median_p_value": 0.15000000000000002,
            "attack_success_rate": {"1e-05": 0.0},
        },
        {
            "token_num": 4,
            "eligible_sample_num": 2,
            "mean_oracle_query_count": 3.5,
            "median_oracle_query_count": 3.5,
            "mean_queries_per_token": 0.875,
            "mean_selected_green_token_count": 1.5,
            "mean_selected_green_ratio": 0.75,
            "median_p_value": 0.0050005,
            "attack_success_rate": {"1e-05": 0.5},
        },
    ]


class VOPRFColorOracleTests(unittest.TestCase):
    def test_blinded_oracle_matches_direct_server_evaluation(self):
        gamma = 0.375
        server = VoprfServer(bytes(range(64)))

        def server_interface(blinded_elements):
            return server.batch_blind_evaluate(blinded_elements)

        oracle = VOPRFColorOracle(
            server_public_key=server.get_public_key(),
            server_interface=server_interface,
            gamma=gamma,
        )
        context = (17, 23, 42)
        token_id = 101

        actual = oracle.query(context, token_id)

        message = struct.pack(">4I", *context, token_id)
        output = server.evaluate(message)
        threshold = int(gamma * (1 << (8 * len(output))))
        expected = int.from_bytes(output, "big") < threshold
        self.assertEqual(actual, expected)
        self.assertEqual(oracle.query_count, 1)

    def test_forged_tokens_and_detector_use_identical_pair_colors(self):
        gamma = 0.5
        seed = bytes(range(64))
        server = VoprfServer(seed)

        def server_interface(blinded_elements):
            return server.batch_blind_evaluate(blinded_elements)

        oracle = VOPRFColorOracle(
            server_public_key=server.get_public_key(),
            server_interface=server_interface,
            gamma=gamma,
        )
        tokenizer = TinyTokenizer()
        forger = AdaptiveWatermarkForger(
            model=TinyCausalModel(),
            tokenizer=tokenizer,
            oracle=oracle,
            window_size=1,
            max_candidates=3,
        )

        forgery = forger.forge("prompt", max_new_tokens=4)
        detection = VOWDetector(
            tokenizer=tokenizer,
            seed=seed,
            gamma=gamma,
            window_size=1,
        ).local_detect(forgery.text)

        unique_selected_pairs = {}
        for step in forgery.steps:
            if step.context is not None:
                unique_selected_pairs[
                    (step.context, step.selected_token_id)
                ] = step.selected_green

        self.assertTrue(forgery.tokenization_preserved)
        self.assertEqual(
            detection.effective_token_num, len(unique_selected_pairs)
        )
        self.assertEqual(
            detection.green_token_num,
            sum(color is True for color in unique_selected_pairs.values()),
        )
        self.assertIsNotNone(forgery.oracle_protocol_stats)
        self.assertEqual(
            forgery.oracle_protocol_stats.query_count,
            forgery.oracle_query_count,
        )
        self.assertGreater(
            forgery.oracle_protocol_stats.total_communication_bytes, 0
        )

        sample_metrics = build_sample_metrics(forgery, detection)
        detection_record = {
            "p_value": detection.p_value,
            "green_token_num": detection.green_token_num,
            "effective_token_num": detection.effective_token_num,
            "green_ratio": detection.green_ratio,
            "total_token_num": detection.total_token_num,
        }
        record = {
            "adaptive_forgery": forgery.to_dict(trace_level="compact"),
            "detection": detection_record,
            "sample_metrics": sample_metrics,
        }
        summary = build_summary(
            SimpleNamespace(
                gamma=gamma,
                max_candidates=3,
                window_size=1,
                significance_levels=[0.01],
            ),
            [record],
            forgery.elapsed_seconds,
        )
        self.assertEqual(
            summary["oracle_query_count"], forgery.oracle_query_count
        )
        self.assertEqual(
            summary["generated_token_num"], forgery.generated_token_num
        )
        self.assertGreater(summary["total_communication_bytes"], 0)


class AdaptiveForgeryTheoryTests(unittest.TestCase):
    def test_truncated_geometric_expectations(self):
        self.assertAlmostEqual(theoretical_green_probability(0.5, 3), 0.875)
        self.assertAlmostEqual(
            theoretical_queries_per_scored_token(0.5, 3), 1.75
        )

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            theoretical_green_probability(0.0, 3)
        with self.assertRaises(ValueError):
            theoretical_green_probability(0.5, 0)


class PerplexityTests(unittest.TestCase):
    def test_uniform_model_has_vocab_size_perplexity(self):
        dataset = Dataset.from_list(
            [
                {"prompt": "1", "target": "2,3"},
                {"prompt": "4", "target": "5"},
            ]
        )

        result = calculate_perplexities(
            model=UniformCausalModel(),
            tokenizer=TinyTokenizer(),
            dataset=dataset,
            prompt_column="prompt",
            target_column="target",
            batch_size=2,
            max_length=16,
            device="cpu",
        )

        self.assertAlmostEqual(result.perplexity, 6.0, places=5)
        self.assertEqual(result.total_target_token_num, 3)
        self.assertEqual(result.sample_target_token_nums, [2, 1])
        for perplexity in result.sample_perplexities:
            self.assertAlmostEqual(perplexity, 6.0, places=5)


if __name__ == "__main__":
    unittest.main()
