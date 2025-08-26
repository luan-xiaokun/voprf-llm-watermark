# Place this near the top of your file, after the imports
import torch
import numpy as np
import time  # Ensure time is imported


class PerformanceMonitor:
    """A utility class to accurately measure LLM generation performance on GPU."""

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("PerformanceMonitor requires a CUDA environment.")
        self.start_event = None
        self.end_event = None
        self.decode_events = []
        self.total_start_time = 0
        self.total_end_time = 0

    def start_total_timer(self):
        """Call this right before the generation process starts."""
        self.total_start_time = time.perf_counter()

    def stop_total_timer(self):
        """Call this right after the generation process ends."""
        self.total_end_time = time.perf_counter()

    def record_step_start(self):
        """Records the start event for a generation step."""
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def record_step_end(self, is_prefill: bool):
        """Records the end event and stores the event pair."""
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.end_event.record()
        if is_prefill:
            self.prefill_events = (self.start_event, self.end_event)
        else:
            self.decode_events.append((self.start_event, self.end_event))

    def calculate_and_print_results(self, num_generated_tokens: int):
        """Calculate and print all performance metrics using amortization."""
        torch.cuda.synchronize()

        print("\n--- PERFORMANCE METRICS (Amortized) ---")

        # --- Prefill Latency (This part remains the same) ---
        prefill_latency_ms = 0
        if self.prefill_events and len(self.prefill_events) == 2:
            prefill_latency_ms = self.prefill_events[0].elapsed_time(
                self.prefill_events[1]
            )
            print(f"Time to First Token (Prefill): {prefill_latency_ms:.2f} ms")

        # --- Amortized Inter-Token Latency (ITL) ---
        avg_amortized_itl_ms = 0
        num_decode_attempts = len(self.decode_events)

        # Calculate the total time spent in all decoding steps (including rejects)
        total_decode_latency_ms = 0
        if self.decode_events:
            total_decode_latency_ms = sum(
                [start.elapsed_time(end) for start, end in self.decode_events]
            )

        # The number of accepted tokens is num_generated_tokens minus the first (prefill) token
        num_accepted_decode_tokens = (
            num_generated_tokens - 1 if num_generated_tokens > 0 else 0
        )

        if num_accepted_decode_tokens > 0:
            avg_amortized_itl_ms = total_decode_latency_ms / num_accepted_decode_tokens

        print(f"--- Inter-Token Latency (ITL) ---")
        print(f"  Total Decode Steps Attempted: {num_decode_attempts}")
        print(f"  Total Decode Tokens Accepted: {num_accepted_decode_tokens}")
        # The rejection rate is a very useful metric to report!
        rejection_rate = (
            (1 - num_accepted_decode_tokens / num_decode_attempts) * 100
            if num_decode_attempts > 0
            else 0
        )
        print(f"  Rejection Rate:               {rejection_rate:.2f}%")
        print(f"  Amortized Average:            {avg_amortized_itl_ms:.2f} ms/token")

        # --- Throughput (TPS) ---
        total_generation_time = self.total_end_time - self.total_start_time
        overall_tps = (
            num_generated_tokens / total_generation_time
            if total_generation_time > 0
            else 0
        )
        # The true decoding TPS is based on the amortized ITL
        decoding_tps = 1000 / avg_amortized_itl_ms if avg_amortized_itl_ms > 0 else 0

        print(f"--- Throughput ---")
        print(f"  Overall TPS (including prefill): {overall_tps:.2f} tokens/sec")
        print(f"  Decoding TPS (from Amortized ITL): {decoding_tps:.2f} tokens/sec")
        print("----------------------------------------\n")
