import json
import math
import random
import time
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers.generation import LogitsProcessor, LogitsProcessorList
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ...utils.io_utils import write_jsonlines
from ..adapter import WatermarkAdapter
from .model_key import BinaryClassifier


def int_to_bin_list(n, number=4):
    return [(n >> (number - 1 - i)) & 1 for i in range(number)]


def get_value(input_x, model):
    device = next(model.parameters()).device
    input_x = input_x.to(device)
    output = model(input_x)
    output = (output > 0.5).bool().item()
    return output


def get_detector_model(input_dim, window_size, model_dir, layers=3):
    model = BinaryClassifier(input_dim, window_size, layers)
    if model_dir is not None:
        state_dict = torch.load(model_dir)
        print("Loaded state dict from", model_dir)
        model.load_state_dict(state_dict)
    return model


class UPVLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab, delta, model, window_size, cache, bit_number, beam_size):
        self.vocab = vocab
        self.delta = delta
        self.model = model
        self.window_size = window_size
        self.cache = cache
        self.bit_number = bit_number
        if beam_size > 0:
            self.beam_size = beam_size
            self.mode = "beam"
        else:
            self.mode = "sample"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

    def _check_green_token(self, input_ids, token_id):
        token_id = int(token_id)
        last_nums = (
            input_ids[-(self.window_size - 1) :] if self.window_size - 1 > 0 else []
        )
        if isinstance(last_nums, torch.Tensor):
            last_nums = last_nums.tolist()

        pair = list(last_nums) + [token_id]
        merged_tuple = tuple(pair)

        if merged_tuple in self.cache:
            return self.cache[merged_tuple]

        bin_list = [int_to_bin_list(num, self.bit_number) for num in pair]
        input_tensor = torch.tensor(bin_list, dtype=torch.float32).unsqueeze(0)

        result = get_value(input_tensor, self.model)
        self.cache[merged_tuple] = result
        return result

    def _rejection_sampling(self, input_ids, scores):
        batch_size = input_ids.shape[0]
        # Current distribution
        probs = torch.softmax(scores, dim=-1)

        # Prepare output scores: mostly -inf to force selection
        new_scores = torch.full_like(scores, -float("inf"))

        for b_idx in range(batch_size):
            while True:
                # 1. Sample from original distribution
                candidate_idx = torch.multinomial(probs[b_idx], 1).item()

                # 2. Check if Green
                is_green = self._check_green_token(input_ids[b_idx], candidate_idx)

                # 3. Accept/Reject
                if is_green:
                    selected = candidate_idx
                    break
                else:
                    # Accept Red with probability 1/e^delta
                    if random.random() < (1.0 / math.exp(self.delta)):
                        selected = candidate_idx
                        break

            # Force the next step to pick this token
            new_scores[b_idx, selected] = 1000.0

        return new_scores

    def _get_greenlist_ids(self, input_ids, scores):
        greenlist_ids = []
        # Get the last 'window_size - 1' items from input_ids
        last_nums = (
            input_ids[-(self.window_size - 1) :] if self.window_size - 1 > 0 else []
        )
        if self.mode == "sample":
            # CRITICAL!
            candidate_tokens = list(range(len(scores)))
        else:
            # Get the score at index 'beam_size'
            threshold_score = torch.topk(
                input=scores, k=self.beam_size, largest=True, sorted=False
            )[0][-1]

            # Get all indices where score is greater than 'score - delta'
            candidate_tokens = (scores >= (threshold_score - self.delta)).nonzero(
                as_tuple=True
            )[0]

        for v in candidate_tokens:
            # Append the current number to the list
            pair = list(last_nums) + [v]
            merged_tuple = tuple(pair)
            bin_list = [int_to_bin_list(num, self.bit_number) for num in pair]

            # load & update cache
            if merged_tuple in self.cache:
                result = self.cache[merged_tuple]
            else:
                result = get_value(torch.FloatTensor(bin_list).unsqueeze(0), self.model)
                self.cache[merged_tuple] = result
            if result:
                greenlist_ids.append(int(v))

        return greenlist_ids

    def _new_get_greenlist_ids(self, input_ids, scores):
        greenlist_ids = []
        # Get the last 'window_size - 1' items from input_ids
        last_nums = (
            input_ids[-(self.window_size - 1) :] if self.window_size - 1 > 0 else []
        )
        if self.mode == "sample":
            _, candidate_tokens = torch.topk(
                input=scores, k=1_000, largest=True, sorted=False
            )
            # candidate_tokens = list(range(len(scores)))
        else:
            # Get the score at index 'beam_size'
            threshold_score = torch.topk(
                input=scores, k=self.beam_size, largest=True, sorted=False
            )[0][-1]
            candidate_tokens = (scores >= (threshold_score - self.delta)).nonzero(
                as_tuple=True
            )[0]

        # Collect tuples that need evaluation
        last_nums_tuple = tuple(last_nums)
        tuples_to_eval = []
        token_indices = []

        for v in candidate_tokens:
            merged_tuple = last_nums_tuple + (int(v),)
            if merged_tuple not in self.cache:
                tuples_to_eval.append(merged_tuple)
                token_indices.append((int(v), len(tuples_to_eval) - 1))
            else:
                if self.cache[merged_tuple]:
                    greenlist_ids.append(int(v))

        # Batch evaluate uncached tuples
        if tuples_to_eval:
            device = next(self.model.parameters()).device

            # Vectorized processing: significantly faster than list comprehension
            tuples_tensor = torch.tensor(
                tuples_to_eval, device=device, dtype=torch.long
            )

            # Create shifts tensor: [bit_number-1, ..., 1, 0]
            shifts = torch.arange(self.bit_number - 1, -1, -1, device=device)

            # Broadcasting to get bits: (N, window_size, 1) >> (1, 1, bit_number) -> (N, window_size, bit_number)
            x = (tuples_tensor.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 1
            x = x.float()

            with torch.no_grad():
                outputs = self.model(x)
                results = (outputs > 0.5).bool().squeeze(-1)
                if results.dim() == 0:
                    results = results.unsqueeze(0)
                results = (
                    results.tolist() if isinstance(results, torch.Tensor) else results
                )

            for tuple_val, result in zip(tuples_to_eval, results):
                self.cache[tuple_val] = result

            for token_id, idx in token_indices:
                if (
                    tuples_to_eval[idx] in self.cache
                    and self.cache[tuples_to_eval[idx]]
                ):
                    greenlist_ids.append(token_id)

        return greenlist_ids

    def _bias_greenlist_logits(
        self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float
    ) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        # if the length of input_id < self.window_size - 1, there is no need to add bias
        if input_ids.shape[-1] < self.window_size - 1:
            # if self.llm_name == "gpt2":
            #     for b_idx in range(input_ids.shape[0]):
            #         scores[b_idx][50256] = -10000
            # elif self.llm_name == "opt-1.3b":
            #     for b_idx in range(input_ids.shape[0]):
            #         scores[b_idx][2] = -10000
            # elif self.llm_name == "llama-7b":
            #     for b_idx in range(input_ids.shape[0]):
            #         scores[b_idx][1] = -10000
            return scores

        if self.mode == "sample":
            return self._rejection_sampling(input_ids, scores)

        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(
                input_ids[b_idx], scores=scores[b_idx]
            )
            green_tokens_mask[b_idx][greenlist_ids] = 1
        green_tokens_mask = green_tokens_mask.bool()

        scores = self._bias_greenlist_logits(
            scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta
        )

        # if self.llm_name == "gpt2":
        #     for b_idx in range(input_ids.shape[0]):
        #         scores[b_idx][50256] = -10000
        # elif self.llm_name == "opt-1.3b":
        #     for b_idx in range(input_ids.shape[0]):
        #         scores[b_idx][2] = -10000
        # elif self.llm_name == "llama-7b":
        #     for b_idx in range(input_ids.shape[0]):
        #         scores[b_idx][1] = -10000

        return scores


class UPVAdapter(WatermarkAdapter):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        detector_dir: str | Path,
        window_size: int,
        delta: float,
        gamma: float = 0.5,
        bit_number: int = 18,
        layers: int = 5,
        beam_size: int = 0,
        timing: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.delta = delta
        self.detector_dir = Path(detector_dir)
        self.gamma = gamma
        self.bit_number = bit_number
        self.layers = layers
        self.beam_size = beam_size
        self.timing = timing

        assert window_size > 0, "Window size must be greater than 0"
        assert delta > 0, "Delta must be greater than 0"
        assert 0 < gamma < 1, "Gamma must be in the range (0, 1)"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.provider_detector_model = get_detector_model(
            bit_number, window_size, self.detector_dir / "combine_model.pt", layers
        ).to(device)
        self.vocab = list(range(1, 2**bit_number - 1))
        self.vocab_size = len(self.vocab)
        self.cache = {}
        self.min_prefix_len = window_size - 1

        self.logits_processor = UPVLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            delta=self.delta,
            model=self.provider_detector_model,
            window_size=self.window_size,
            cache=self.cache,
            bit_number=self.bit_number,
            beam_size=self.beam_size,
        )

    def get_logits_processor(
        self, do_sample: bool, num_beams: int, top_k: float | None
    ) -> LogitsProcessorList | None:
        return self.logits_processor

    def judge_green(self, input_ids, current_number):
        # Get the last 'window_size - 1' items from input_ids
        last_nums = (
            input_ids[-(self.window_size - 1) :] if self.window_size - 1 > 0 else []
        )
        # Append the current number to the list
        pair = list(last_nums) + [current_number]
        merged_tuple = tuple(pair)
        bin_list = [int_to_bin_list(num, self.bit_number) for num in pair]
        # merged_list = sum(bin_list, [])

        # load & update cache
        if merged_tuple in self.cache:
            result = self.cache[merged_tuple]
        else:
            result = get_value(
                torch.FloatTensor(bin_list).unsqueeze(0), self.provider_detector_model
            )
            self.cache[merged_tuple] = result

        return result

    def random_sample(self, input_ids, is_green):
        # Get the last 'window_size - 1' items from input_ids
        last_nums = (
            input_ids[-(self.window_size - 1) :] if self.window_size - 1 > 0 else []
        )
        while True:
            number = random.choice(self.vocab)
            # Append the new random number to the list
            pair = list(last_nums) + [number]
            merged_tuple = tuple(pair)
            bin_list = [int_to_bin_list(num, self.bit_number) for num in pair]

            if merged_tuple in self.cache:
                result = self.cache[merged_tuple]
            else:
                result = get_value(
                    torch.FloatTensor(bin_list).unsqueeze(0),
                    self.provider_detector_model,
                )
                self.cache[merged_tuple] = result

            if is_green and result:
                return number

            elif not is_green and not result:
                return number

    def generate_list_with_green_ratio(self, length: int, green_ratio: float):
        token_list = random.sample(self.vocab, self.window_size - 1)
        is_green = []

        while len(token_list) < length:
            green = 1 if random.random() < green_ratio else 0
            if green:
                token = self.random_sample(torch.LongTensor(token_list), True)
                token_list.append(token)
                is_green.append(1)
            else:
                token = self.random_sample(torch.LongTensor(token_list), False)
                token_list.append(token)
                is_green.append(0)

        # loop
        is_green_append = []
        for i in range(0, self.window_size - 1):
            tail_slice = token_list[-(self.window_size - 1 - i) :]
            head_slice = token_list[:i]
            input_slice = tail_slice + head_slice
            is_green_append.append(self.judge_green(input_slice, token_list[i]))

        is_green = is_green_append + is_green

        return token_list, is_green

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        sigma = 0.01
        expected_count = self.gamma
        number = observed_count - expected_count * T
        denom = math.sqrt(T * expected_count * (1 - expected_count) + sigma * sigma * T)
        z = number / denom
        return z

    def green_token_mask_and_stats(self, input_ids: torch.Tensor):
        mask_list = []
        green_token_count = 0
        for idx in range(self.min_prefix_len, len(input_ids)):
            curr_token = input_ids[idx]
            if self.judge_green(input_ids[:idx], curr_token):
                mask_list.append(True)
                green_token_count += 1
            else:
                mask_list.append(False)
        num_tokens_scored = len(input_ids) - self.min_prefix_len
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return mask_list, green_token_count, z_score

    def batch_green_token_z_scores(self, batch_input_ids) -> torch.Tensor:
        """
        Efficient batch version that computes only z-scores for a batch of input_id sequences.
        Accepts:
          - List[List[int]] or List[torch.Tensor] of variable lengths, or
          - torch.LongTensor of shape (batch, seq_len) (assumed no padding or already trimmed).
        Returns:
          - torch.FloatTensor of shape (batch,) containing per-sequence z-scores.
        """
        # Normalize inputs into list of python lists (variable lengths supported)
        if isinstance(batch_input_ids, torch.Tensor):
            if batch_input_ids.ndim == 1:
                seqs = [batch_input_ids.tolist()]
            elif batch_input_ids.ndim == 2:
                seqs = [row.tolist() for row in batch_input_ids]
            else:
                raise ValueError("batch_input_ids must be 1D or 2D tensor.")
        elif isinstance(batch_input_ids, list):
            seqs = []
            for s in batch_input_ids:
                if isinstance(s, torch.Tensor):
                    seqs.append(s.tolist())
                else:
                    seqs.append(list(s))
        else:
            raise ValueError("Unsupported type for batch_input_ids.")

        B = len(seqs)
        if B == 0:
            return torch.empty(0, dtype=torch.float32)

        ws = self.window_size
        pref = self.min_prefix_len  # ws - 1

        # 1) Collect all window keys (tuples) that need to be evaluated, deduped and cache-aware
        new_keys = []
        # For mapping each sequence index to its list of keys for fast counting later
        per_seq_keys = [[] for _ in range(B)]

        for b_idx, seq in enumerate(seqs):
            L = len(seq)
            if L <= pref:
                # nothing to score
                continue
            # Iterate scored token indices
            for idx in range(pref, L):
                # Build window tuple: last (ws-1) tokens before idx, plus current token
                # This avoids slicing [:idx] then taking the tail.
                start = idx - (ws - 1) if ws > 1 else idx
                last_nums = seq[start:idx] if ws > 1 else []
                key = tuple(last_nums + [seq[idx]])
                per_seq_keys[b_idx].append(key)
                if key not in self.cache:
                    new_keys.append(key)

        # Deduplicate new keys
        if new_keys:
            # Keep order not required; set is fine
            unique_new_keys = list(set(new_keys))
        else:
            unique_new_keys = []

        # 2) Batch-evaluate the detector model for uncached keys
        if unique_new_keys:
            # Build bit tensors for all unique_new_keys
            bit_cache: dict[int, list[int]] = {}

            def token_bits(t: int) -> list[int]:
                v = bit_cache.get(t)
                if v is None:
                    # Most significant bit first
                    v = [(t >> k) & 1 for k in range(self.bit_number - 1, -1, -1)]
                    bit_cache[t] = v
                return v

            bits_batches = []
            for key in unique_new_keys:
                # key is length=ws tuple of ints
                bits = [token_bits(tok) for tok in key]  # ws x bit_number
                bits_batches.append(bits)

            x = torch.tensor(bits_batches, dtype=torch.float32)
            # Shape expected by model: (N, ws, bit_number)
            device = next(self.provider_detector_model.parameters()).device
            x = x.to(device)

            self.provider_detector_model.eval()
            with torch.no_grad():
                out = self.provider_detector_model(x)
                # Expect shape (N,) or (N,1)
                out = out.squeeze(-1)
                preds = (out > 0.5).to(torch.bool).tolist()

            # Update cache with results
            for k, p in zip(unique_new_keys, preds):
                self.cache[k] = p

        # 3) Count greens per sequence using cache, compute z-scores vectorized
        counts = []
        totals = []
        for b_idx, seq in enumerate(seqs):
            keys = per_seq_keys[b_idx]
            green_count = 0
            for k in keys:
                green_count += 1 if self.cache.get(k, False) else 0
            counts.append(green_count)
            totals.append(max(0, len(seq) - pref))

        counts_t = torch.tensor(counts, dtype=torch.float32)
        totals_t = torch.tensor(totals, dtype=torch.float32)

        gamma = float(self.gamma)
        sigma = 0.01
        denom = torch.sqrt(
            totals_t * gamma * (1.0 - gamma) + (sigma * sigma) * totals_t
        )
        # Avoid division by zero when totals_t == 0
        z = torch.where(
            totals_t > 0,
            (counts_t - gamma * totals_t) / torch.clamp(denom, min=1e-12),
            torch.zeros_like(totals_t),
        )
        return z

    def generate_and_save_train_data(self, num_samples: int, output_dir: Path | str):
        train_data = []
        for _ in tqdm(range(num_samples)):
            length = 200
            green_ratio = random.random()
            token_list, is_green = self.generate_list_with_green_ratio(
                length, green_ratio
            )
            _, _, z_score = self.green_token_mask_and_stats(torch.tensor(token_list))
            train_data.append((tuple(token_list), tuple(is_green), z_score))

        train_data = list(set(train_data))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "train_data.jsonl", "w") as f:
            for item in train_data:
                json.dump(
                    {
                        "Input": [int(i) for i in item[0]],
                        "Tag": [int(i) for i in item[1]],
                        "Output": float(item[2]),
                    },
                    f,
                )
                f.write("\n")
        print(f"Saved training data to {output_dir / 'train_data.jsonl'}")

    def generate_and_save_test_data(
        self, dataset: Dataset, output_dir, sampling_temp, max_new_tokens
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        watermark_processor = UPVLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            delta=self.delta,
            model=self.provider_detector_model,
            window_size=self.window_size,
            cache=self.cache,
            bit_number=self.bit_number,
            beam_size=self.beam_size,
        )

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_k": 20,
            "temperature": sampling_temp,
        }

        self.model.to(device)
        generate_with_watermark = partial(
            self.model.generate,
            logits_processor=LogitsProcessorList([watermark_processor]),
            no_repeat_ngram_size=4,
            **gen_kwargs,
        )

        num = 500
        prompt_column = "prompt_text"
        max_non_watermarked_tokens = 200

        for sample in tqdm(dataset.select(range(num))):
            text = sample[prompt_column]
            text_tokenized = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=True
            ).to(device)

            prompt = {
                "input_ids": text_tokenized["input_ids"],
                "attention_mask": text_tokenized["attention_mask"],
            }

            output_with_watermark = generate_with_watermark(**prompt)
            output_with_watermark = output_with_watermark[
                :, prompt["input_ids"].shape[-1] :
            ]

            _, _, z_score1 = self.green_token_mask_and_stats(
                output_with_watermark.squeeze(0)
            )
            item = {
                "Input": self.tokenizer.batch_decode(
                    output_with_watermark, skip_special_tokens=True
                )[0],
                "Tag": 1,
                "Z-score": z_score1,
                "Type": "Watermarked",
            }
            write_jsonlines(output_dir / "test_data.jsonl", item, mode="a")

            completion_tokens = self.tokenizer(
                sample["completion_text"],
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"][0]

            truncated_completion_tokens = completion_tokens[:max_non_watermarked_tokens]

            _, _, z_score2 = self.green_token_mask_and_stats(
                truncated_completion_tokens
            )

            item = {
                "Input": self.tokenizer.decode(
                    truncated_completion_tokens, skip_special_tokens=True
                ),
                "Tag": 0,
                "Z-score": z_score2,
                "Type": "Non-watermarked",
            }
            write_jsonlines(output_dir / "test_data.jsonl", item, mode="a")
            print(
                f"Watermarked Z-score: {z_score1}, Non-watermarked Z-score: {z_score2}"
            )
