import inspect

import numpy as np
import torch
from transformers.generation import (
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import set_seed


class GpuTimingLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that accurately records the GPU execution time for each
    generation step using torch.cuda.Event, and applies a nested
    watermarking processor.
    """

    def __init__(self, watermark_processor: LogitsProcessor | None = None):
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("This timing processor requires a CUDA environment.")

        if watermark_processor is None:
            watermark_processor = lambda x, y: y

        self.watermark_processor = watermark_processor

        # We will store a list of CUDA events, one for each step
        self.events = []

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Create a new CUDA event and record a timestamp in the default CUDA stream
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events.append(event)

        return self.watermark_processor(input_ids, scores)

    def get_results(self) -> dict:
        """
        Computes and returns the timing results. This method MUST be called
        AFTER model.generate() has finished.
        """
        # We must synchronize here to ensure all recorded events have been processed
        # by the GPU. This is a one-time cost at the end of generation.
        torch.cuda.synchronize()

        if len(self.events) < 2:
            return {
                "avg_inter_token_latency_ms": 0,
                "p95_inter_token_latency_ms": 0,
                "num_decoded_tokens": 0,
            }

        # Calculate the time elapsed between consecutive events
        latencies_ms = [
            self.events[i - 1].elapsed_time(self.events[i])
            for i in range(1, len(self.events))
        ]

        # Calculate statistics
        avg_itl = np.mean(latencies_ms)
        p95_itl = np.percentile(latencies_ms, 95)

        return {
            "avg_inter_token_latency_ms": avg_itl,
            "p95_inter_token_latency_ms": p95_itl,
            "num_decoded_tokens": len(latencies_ms),
        }


class WatermarkAdapter:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    seed: int
    _timing_logits_processor: GpuTimingLogitsProcessor
    timing: bool = False

    def get_logits_processor(
        self, do_sample: bool, num_beams: int, top_k: float | None
    ) -> LogitsProcessor | None:
        pass

    def get_generation_config(
        self,
        max_new_tokens: int | None,
        stop_strings: str | list[str] | None,
        do_sample: bool,
        num_beams: int,
        top_p: float | None,
        top_k: int | None,
        temperature: float | None,
        suppress_tokens: list[int] | None,
        no_watermark: bool,
    ) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            suppress_tokens=suppress_tokens,
        )

    def _pre_generation(
        self,
        do_sample: bool,
        num_beams: int,
        top_p: float | None,
        top_k: float | None,
        no_watermark: bool,
    ) -> None:
        pass

    def _post_generation(self) -> None:
        pass

    def __call__(
        self,
        prompts: str | list[str],
        num_samples: int = 1,
        max_new_tokens: int | None = None,
        stop_strings: str | list[str] | None = None,
        seed: int | None = None,
        do_sample: bool = False,
        num_beams: int = 1,
        top_p: float | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        suppress_tokens: list[int] | None = None,
        no_watermark: bool = False,
        **model_specific_params,
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
        if num_samples > 1:
            inputs["input_ids"] = inputs["input_ids"].repeat_interleave(
                num_samples, dim=0
            )
            inputs["attention_mask"] = inputs["attention_mask"].repeat_interleave(
                num_samples, dim=0
            )

        if (
            "attention_mask"
            not in inspect.signature(self.model.forward).parameters.keys()
        ):
            del inputs["attention_mask"]

        if seed is not None:
            set_seed(seed)

        # pre generation processing
        # e.g., check decode method, set up sampling parameters
        self._pre_generation(do_sample, num_beams, top_p, top_k, no_watermark)

        logits_processor = None
        if not no_watermark:
            logits_processor = self.get_logits_processor(do_sample, num_beams, top_k)
        timing_logits_processor = GpuTimingLogitsProcessor(logits_processor)

        logits_processor_list = LogitsProcessorList([timing_logits_processor])

        generation_config = self.get_generation_config(
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            do_sample=do_sample,
            num_beams=num_beams,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            suppress_tokens=suppress_tokens,
            no_watermark=no_watermark,
        )

        output_ids = self.model.generate(
            **inputs,
            generation_config=generation_config,
            tokenizer=self.tokenizer,
            logits_processor=logits_processor_list,
            **model_specific_params,
        )

        if self.timing and timing_logits_processor is not None:
            print(timing_logits_processor.get_results())

        # post generation processing
        # e.g., replacing the hacked _sample method with the original one
        self._post_generation()

        if self.model.config.is_encoder_decoder:
            generated_ids = output_ids
        else:
            generated_ids = output_ids[:, input_ids.shape[1] :]

        if num_samples > 1 and isinstance(prompts, list):
            generated_ids = generated_ids.view(input_ids.size(0), num_samples, -1)

        if isinstance(prompts, str):
            generated_ids = generated_ids.squeeze(0)

        return self._decode_generation(generated_ids)

    def _decode_generation(self, generated_ids: torch.LongTensor) -> str | list[str]:
        if generated_ids.dim() == 1:
            return self.tokenizer.decode(
                generated_ids.tolist(), skip_special_tokens=True
            )
        if generated_ids.dim() == 2:
            return [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in generated_ids.tolist()
            ]
        if generated_ids.dim() == 3:
            return [
                self.tokenizer.decode(
                    generated_ids[i].tolist(), skip_special_tokens=True
                )
                for i in range(len(generated_ids))
            ]
        raise TypeError(
            f"Generated outputs aren't 1D, 2D or 3D, but instead are {generated_ids.shape}"
        )

    def __getattr__(self, name: str):
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
