import heapq
import inspect
import random
import secrets
import struct
from typing import NamedTuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import (
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
)
from voprf_py import VoprfServer


def dummy_process_logits(p: float):
    return random.random() < p


class VoprfCallRecord(NamedTuple):
    vocab_size: int
    candidate_size: int
    num_calls: int


class TopkAdaptiveLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        window_size: int,
        delta: float,
        gamma: float,
        top_k: int,
        voprf_server: VoprfServer,
        mini_batch_size: int | None = None,
    ) -> None:
        self.window_size = window_size
        self.delta = delta
        self.gamma = gamma
        self.top_k = top_k
        self.call_records: list[VoprfCallRecord] = []
        self.voprf_server = voprf_server
        self.mini_batch_size = mini_batch_size or top_k

    @torch.no_grad()
    def __bak_call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if scores.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            scores = scores.unsqueeze(0)
        elif scores.dim() != 2:
            raise ValueError(f"Unsupported scores dimension: {scores.dim()}")

        device = input_ids.device
        batch_size, vocab_size = scores.shape
        contexts = input_ids[:, -self.window_size :].tolist()
        top_k_scores, _ = scores.topk(self.top_k)
        cutoff_values = top_k_scores[:, -1] - self.delta

        for i in range(batch_size):
            context = contexts[i]
            context_bytes = struct.pack(f"{len(context)}I", *context)

            sub_scores = scores[i]
            candidate_mask = sub_scores >= cutoff_values[i]
            candidate_indices = torch.nonzero(candidate_mask).squeeze(-1)
            candidate_size = candidate_indices.numel()

            candidate_scores = sub_scores[candidate_indices]
            sorted_candidates = sorted(
                zip(candidate_scores.tolist(), candidate_indices.tolist()), reverse=True
            )

            num_calls = 0
            top_k_heap = []

            # mini-batch size set to self.top_k
            mini_batch_size = self.top_k
            for j in range(0, candidate_size, mini_batch_size):
                size = min(mini_batch_size, candidate_size - j)
                mini_batch = sorted_candidates[j : j + size]

                msg_inputs = [
                    context_bytes + struct.pack(">I", token_id)
                    for _, token_id in mini_batch
                ]
                msg_hashes = self.voprf_server.batch_evaluate(msg_inputs)
                probs = [
                    int.from_bytes(msg_hash, "big") / (1 << 8 * len(msg_hash))
                    for msg_hash in msg_hashes
                ]
                num_calls += len(msg_hashes)

                for prob, (score, token_id) in zip(probs, mini_batch):
                    if prob < self.gamma:
                        new_score = score + self.delta
                    else:
                        new_score = score

                    if len(top_k_heap) < self.top_k:
                        heapq.heappush(top_k_heap, (new_score, token_id))
                    else:
                        heapq.heappushpop(top_k_heap, (new_score, token_id))

                if j + size < candidate_size and len(top_k_heap) >= self.top_k:
                    next_score, _ = sorted_candidates[j + size]
                    kth_score = top_k_heap[0][0]
                    if kth_score >= next_score + self.delta:
                        break

            # for j, (score, token_id) in enumerate(sorted_candidates):
            #     msg_input = context_bytes + struct.pack(">I", token_id)
            #     msg_hash = self.server.evaluate(msg_input)
            #     prob = int.from_bytes(msg_hash, "big") / (1 << 8 * len(msg_hash))
            #     num_calls += 1

            #     if prob < self.p:
            #         new_score = score + self.delta
            #     else:
            #         new_score = score

            #     if len(top_k_heap) < self.top_k:
            #         heapq.heappush(top_k_heap, (new_score, token_id))
            #     else:
            #         heapq.heappushpop(top_k_heap, (new_score, token_id))

            #     if j + 1 < candidate_size and len(top_k_heap) >= self.top_k:
            #         next_score, _ = sorted_candidates[j + 1]
            #         kth_score = top_k_heap[0][0]
            #         if kth_score >= next_score + self.delta:
            #             break

            if top_k_heap:
                top_k_scores, top_k_indices = zip(*top_k_heap)
                top_k_indices = torch.tensor(top_k_indices, device=device)
                scores[i][top_k_indices] = torch.tensor(top_k_scores, device=device)

            self.call_records.append(
                VoprfCallRecord(vocab_size, candidate_size, num_calls)
            )

        return scores

    @torch.no_grad()
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if scores.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            scores = scores.unsqueeze(0)
        elif scores.dim() != 2:
            raise ValueError(f"Unsupported scores dimension: {scores.dim()}")

        device = input_ids.device
        batch_size, vocab_size = scores.shape

        # tried my best to make this code more efficient

        # Step 1. prepare for candidate selection and padding

        # 1.1 cutoff values
        top_k_for_cutoff, _ = scores.topk(self.top_k, dim=1)
        cutoff_values = top_k_for_cutoff[:, -1] - self.delta

        # 1.2 candidate mask
        candidate_mask = scores >= cutoff_values.unsqueeze(1)

        # 1.3 pad jagged candidates
        context_bytes_list = [
            struct.pack(f">{len(ctx)}I", *ctx)
            for ctx in input_ids[:, -self.window_size :].tolist()
        ]
        num_candidates_per_sample = candidate_mask.sum(dim=1)
        max_candidate_size = num_candidates_per_sample.max().item()

        padded_indices = torch.full(
            (batch_size, max_candidate_size),
            -1,
            dtype=torch.long,
            device=device,
        )
        padded_scores = torch.full(
            (batch_size, max_candidate_size),
            -float("inf"),
            dtype=torch.float,
            device=device,
        )

        for i in range(batch_size):
            n_candidates = num_candidates_per_sample[i]
            sample_indices = torch.nonzero(candidate_mask[i]).squeeze(-1)
            padded_indices[i, :n_candidates] = sample_indices
            padded_scores[i, :n_candidates] = scores[i, sample_indices]

        # 1.4 sort candidates in descending order
        sorted_scores, sorted_indices = torch.sort(
            padded_scores, dim=1, descending=True
        )
        sorted_indices = torch.gather(padded_indices, 1, sorted_indices)

        # Step 2. tensor preparation for top-k heaps and active mask

        # 2.1 tensor for all top-k heaps
        top_k_scores_batch = torch.full(
            (batch_size, self.top_k), -float("inf"), device=device
        )
        top_k_indices_batch = torch.full(
            (batch_size, self.top_k), -1, dtype=torch.long, device=device
        )

        # 2.2 active mask to track which samples still have candidates
        active_mask = num_candidates_per_sample > 0

        total_calls = 0

        # Step 3. batch processing of candidates and early stopping

        # main loop, each iteration for a mini-batch of candidates
        for j in range(0, max_candidate_size, self.mini_batch_size):
            if not active_mask.any():
                break

            # 3.1 slice for current mini-batch
            size = min(self.mini_batch_size, max_candidate_size - j)
            cur_indices_slice = sorted_indices[:, j : j + size]
            cur_scores_slice = sorted_scores[:, j : j + size]

            active_indices_gpu = torch.nonzero(active_mask).squeeze(-1)

            indices_cpu = cur_indices_slice[active_indices_gpu].cpu()
            scores_cpu = cur_scores_slice[active_indices_gpu].cpu()
            active_indices_cpu = active_indices_gpu.cpu().tolist()

            msg_inputs = []
            msg_metadata = []

            for i_idx, batch_idx in enumerate(active_indices_cpu):
                context_bytes = context_bytes_list[batch_idx]
                for k in range(self.mini_batch_size):
                    token_id = indices_cpu[i_idx, k]  # on device
                    if token_id != -1:
                        msg_inputs.append(
                            context_bytes + struct.pack(">I", token_id.item())
                        )
                        msg_metadata.append(
                            (batch_idx, k, scores_cpu[i_idx, k], token_id)
                        )

            if not msg_inputs:
                continue

            # 3.2 evaluate the whole mini-batch in one call
            msg_hashes = self.voprf_server.batch_evaluate(msg_inputs)
            total_calls += len(msg_hashes)

            # 3.3 process the results
            new_scores_list = []
            batch_indices_list = []
            k_indices_list = []
            token_ids_list = []

            for idx, msg_hash in enumerate(msg_hashes):
                prob = int.from_bytes(msg_hash, "big") / (1 << 8 * len(msg_hash))
                batch_idx, k_idx, original_score, token_id = msg_metadata[idx]

                new_score = original_score + self.delta * (prob < self.gamma)

                new_scores_list.append(new_score)
                batch_indices_list.append(batch_idx)
                k_indices_list.append(k_idx)
                token_ids_list.append(token_id)

            upd_batch_indices = torch.tensor(
                batch_indices_list, device=device, dtype=torch.long
            )
            upd_k_indices = torch.tensor(
                k_indices_list, device=device, dtype=torch.long
            )
            upd_new_scores = torch.tensor(
                new_scores_list, device=device, dtype=torch.float
            )
            upd_token_ids = torch.tensor(
                token_ids_list, device=device, dtype=torch.long
            )

            new_scores_batch = torch.full(
                (batch_size, self.mini_batch_size), -float("inf"), device=device
            )
            new_indices_batch = torch.full(
                (batch_size, self.mini_batch_size), -1, dtype=torch.long, device=device
            )
            new_scores_batch[upd_batch_indices, upd_k_indices] = upd_new_scores
            new_indices_batch[upd_batch_indices, upd_k_indices] = upd_token_ids

            # 3.4 update top-k heaps
            combined_scores = torch.cat([top_k_scores_batch, new_scores_batch], dim=1)
            combined_indices = torch.cat(
                [top_k_indices_batch, new_indices_batch], dim=1
            )

            sorted_combined_scores, best_indices_in_combined = torch.topk(
                combined_scores, self.top_k, dim=1
            )

            top_k_scores_batch = sorted_combined_scores
            top_k_indices_batch = torch.gather(
                combined_indices, 1, best_indices_in_combined
            )

            # 3.5 check termination condition
            kth_scores = top_k_scores_batch[:, -1]
            next_candidate_start_idx = j + size
            if next_candidate_start_idx < max_candidate_size:
                next_scores = sorted_scores[:, next_candidate_start_idx]
                stop_condition = kth_scores >= next_scores + self.delta
                active_mask &= ~stop_condition

            active_mask &= next_candidate_start_idx < num_candidates_per_sample

        # record statistics
        avg_candidate_size = num_candidates_per_sample.float().mean().item()
        self.call_records.append(
            VoprfCallRecord(vocab_size, avg_candidate_size, total_calls)
        )

        # Step 4. finalize scores

        final_scores = scores.clone()

        valid_final_mask = top_k_indices_batch != -1
        row_indices = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand_as(top_k_indices_batch)
        )

        final_scores[
            row_indices[valid_final_mask], top_k_indices_batch[valid_final_mask]
        ] = top_k_scores_batch[valid_final_mask]

        return final_scores


class TestAdapter:
    def __init__(
        self, model, tokenizer, delta: float, gamma: float, seed: bytes
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.delta = delta
        self.gamma = gamma
        self.seed = seed

    def __call__(
        self, prompts: str, top_k: int = 10, num_samples: int = 1
    ) -> str | list[str]:
        prompts_ = prompts
        if isinstance(prompts, str):
            prompts_ = [prompts]

        batch_encoding = self.tokenizer(prompts_, return_tensors="pt", padding=True)
        input_ids: torch.LongTensor = batch_encoding["input_ids"]
        attention_mask: torch.LongTensor = batch_encoding["attention_mask"]
        inputs = {
            "input_ids": input_ids.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device),
        }
        if (
            "attention_mask"
            not in inspect.signature(self.model.forward).parameters.keys()
        ):
            del inputs["attention_mask"]

        generation_config = GenerationConfig(
            max_new_tokens=200,
            num_return_sequences=num_samples,
            top_k=top_k,
            do_sample=True,
        )
        logits_processor = LogitsProcessorList(
            [
                TopkAdaptiveLogitsProcessor(
                    window_size=7,
                    delta=self.delta,
                    gamma=self.gamma,
                    top_k=top_k,
                    voprf_server=VoprfServer(self.seed),
                )
            ]
        )

        output_ids = self.model.generate(
            **inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
        )

        if self.model.config.is_encoder_decoder:
            generated_ids = output_ids
        else:
            generated_ids = output_ids[:, input_ids.shape[1] :]

        if num_samples > 1 and isinstance(prompts, list):
            generated_ids = generated_ids.view(input_ids.size(0), num_samples, -1)

        if isinstance(prompts, str):
            generated_ids = generated_ids.squeeze(0)

        # output call statistics
        processor: TopkAdaptiveLogitsProcessor = logits_processor[0]
        avg_vocab_size = sum(
            record.vocab_size for record in processor.call_records
        ) / max(1, len(processor.call_records))
        avg_candidate_size = sum(
            record.candidate_size for record in processor.call_records
        ) / max(1, len(processor.call_records))
        avg_num_calls = sum(
            record.num_calls for record in processor.call_records
        ) / max(1, len(processor.call_records))
        print(
            f"Average Vocab Size: {avg_vocab_size}, "
            f"Average Candidate Size: {avg_candidate_size}, "
            f"Average Num Calls: {avg_num_calls}"
        )

        return self._decode_generation(generated_ids)

    def _decode_generation(self, generated_ids: torch.LongTensor) -> str | list[str]:
        if generated_ids.dim() == 1:
            return self.tokenizer.decode(generated_ids.tolist())
        if generated_ids.dim() == 2:
            return [self.tokenizer.decode(ids) for ids in generated_ids.tolist()]
        if generated_ids.dim() == 3:
            return [
                self.tokenizer.decode(generated_ids[i].tolist())
                for i in range(len(generated_ids))
            ]
        raise TypeError(
            f"Generated outputs aren't 1D, 2D or 3D, but instead are {generated_ids.shape}"
        )


def main():
    import time
    from detection import detect

    seed = secrets.token_bytes(32)
    model_path = "models/Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    adapter = TestAdapter(
        model=model, tokenizer=tokenizer, delta=2.0, gamma=0.5, seed=seed
    )

    generation_total_time = time.time()
    outputs = adapter(
        prompts="I have a dream",
        top_k=100,
        num_samples=128,
    )
    generation_total_time = time.time() - generation_total_time
    generated_token_num = 0
    for i, output in enumerate(outputs):
        generated_token_num += len(tokenizer.encode(output, add_special_tokens=False))
        print(f"Sample {i + 1}: {output}")
    avg_tps = generated_token_num / generation_total_time
    print(f"Average Tokens Per Second: {avg_tps:.2f}")

    voprf_server = VoprfServer(seed)
    green_count = 0
    for output in outputs:
        tokens = tokenizer.encode(output, add_special_tokens=False)
        for i in range(7, len(tokens)):
            context = tokens[i - 7 : i]
            token = tokens[i]
            context_bytes = struct.pack(f">{len(context)}I", *context)
            msg_input = context_bytes + struct.pack(">I", token)
            msg_hash = voprf_server.evaluate(msg_input)
            sampling_prob = int.from_bytes(msg_hash, "big") / (1 << 8 * len(msg_hash))
            is_green = sampling_prob < 0.5
            green_count += is_green
    print(
        f"Green tokens: {green_count} / {generated_token_num} ({green_count / generated_token_num:.2%})"
    )


if __name__ == "__main__":
    main()
