import struct
from typing import NamedTuple

import torch
from transformers.generation import LogitsProcessor
from voprf_py import VoprfServer


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
                for k in range(size):
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

        final_scores = torch.full_like(scores, -float("inf"))

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
