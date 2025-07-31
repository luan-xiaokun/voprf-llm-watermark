import math
import os
import struct
import types
from typing import TYPE_CHECKING, Optional, Union

import torch
from torch import nn
from transformers import logging
from transformers.generation.configuration_utils import CompileConfig, GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
)
from transformers.modeling_utils import PreTrainedModel
from voprf_py import VoprfServer

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)

# Equivalent classes (kept for retrocompatibility purposes)
GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
ContrastiveSearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput

ContrastiveSearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
GreedySearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput

BeamSearchDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput
BeamSampleDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput

BeamSearchEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput
BeamSampleEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput

GreedySearchOutput = Union[
    GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput
]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[
    ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput
]

# Typing shortcuts
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[
    GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput
]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]


# class RejectionSamplingMixin(GenerationMixin):
#     window_size: int
#     delta: float
#     gamma: float
#     voprf_server: VoprfServer


def rejection_sampling(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    A rejection sampling generation method based on transformers' `GenerationMixin`.

    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(
        cur_len, input_ids.device, model_kwargs
    )

    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        # If we use FA2 and a static cache, we cannot compile with fullgraph
        if self.config._attn_implementation == "flash_attention_2" and getattr(
            model_kwargs.get("past_key_values"), "is_compileable", False
        ):
            if generation_config.compile_config is None:
                generation_config.compile_config = CompileConfig(fullgraph=False)
            # only raise warning if the user passed an explicit compile-config (otherwise, simply change the default without confusing the user)
            elif generation_config.compile_config.fullgraph:
                logger.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(
            input_ids, generation_config, **model_kwargs
        )
        is_prefill = False
    else:
        is_prefill = True

    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=input_ids.device
        )

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # ==================================
        # here we go for rejection sampling!
        # ==================================
        # next_tokens: (batch_size,)
        # input_ids: (batch_size, cur_len)
        active_mask = unfinished_sequences.bool()
        active_indices = active_mask.nonzero().squeeze(-1)

        if active_indices.numel() != 0:
            active_next_tokens = next_tokens[active_mask]
            active_input_ids = input_ids[active_mask]
            active_probs = probs[active_mask]

            batch_size = input_ids.shape[0]

            active_context_bytes = [
                struct.pack(f">{len(ctx)}I", *ctx)
                for ctx in active_input_ids[:, -self.window_size :].tolist()
            ]
            rejection_mask = _get_rejection_mask(
                active_next_tokens,
                active_context_bytes,
                self.delta,
                self.gamma,
                self.voprf_server,
            )

            while rejection_mask.any():
                rejected_in_active = rejection_mask.nonzero().squeeze(-1)

                resampled_probs = active_probs[rejection_mask]

                rejected_indices = rejection_mask.nonzero().squeeze(-1)
                rejected_contexts = [
                    active_context_bytes[i] for i in rejected_in_active
                ]

                resampled_next_tokens = torch.multinomial(
                    resampled_probs, num_samples=1
                ).squeeze(1)

                new_rejection_mask = _get_rejection_mask(
                    resampled_next_tokens,
                    rejected_contexts,
                    self.delta,
                    self.gamma,
                    self.voprf_server,
                )

                acc_mask_in_resamples = ~new_rejection_mask
                indices_to_update_in_active = rejected_indices[acc_mask_in_resamples]
                tokens_to_place = resampled_next_tokens[acc_mask_in_resamples]
                active_next_tokens[indices_to_update_in_active] = tokens_to_place

                rejection_mask[rejection_mask.clone()] = new_rejection_mask

            next_tokens[active_mask] = active_next_tokens
        # =========================
        # end of rejection sampling
        # =========================

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def _get_rejection_mask(
    tokens_to_check: torch.Tensor,
    contexts: list[bytes],
    delta: float,
    gamma: float,
    voprf_server: VoprfServer,
) -> torch.Tensor:
    device = tokens_to_check.device
    w_max = math.exp(delta)
    msg_inputs = [
        ctx_bytes + struct.pack(">I", token_id)
        for ctx_bytes, token_id in zip(contexts, tokens_to_check.tolist())
    ]
    msg_hashes = voprf_server.batch_evaluate(msg_inputs)
    sampling_probs = torch.tensor(
        [int.from_bytes(h, "big") / (1 << 8 * len(h)) for h in msg_hashes],
        device=device,
    )

    weights = torch.where(sampling_probs < gamma, w_max, 1.0)
    mask = w_max * torch.rand(len(tokens_to_check), device=device) >= weights

    return mask


def replace_with_rejection_sampling_mixin(
    model: PreTrainedModel,
    window_size: int,
    delta: float,
    gamma: float,
    voprf_server: VoprfServer,
):
    model.window_size = window_size
    model.delta = delta
    model.gamma = gamma
    model.voprf_server = voprf_server

    original_sample = model._sample

    model._sample = types.MethodType(rejection_sampling, model)

    return original_sample


def recover_sampling_mixin(model: PreTrainedModel, original_sample):
    if not hasattr(model, "window_size"):
        raise ValueError("Model does not have a rejection sampling mixin applied.")

    model._sample = types.MethodType(original_sample, model)

    del model.window_size
    del model.delta
    del model.gamma
    del model.voprf_server

    return model


def main():
    import inspect
    import secrets
    import time

    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed = secrets.token_bytes(32)
    window_size = 7
    model_path = "models/Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    original_sample = replace_with_rejection_sampling_mixin(
        model=model, window_size=window_size, delta=3.0, gamma=0.5, seed=seed
    )
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    prompts = ["Tell me a story about a"]
    num_samples = 128

    generation_total_time = time.time()
    batch_encoding = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids: torch.LongTensor = batch_encoding["input_ids"]
    attention_mask: torch.LongTensor = batch_encoding["attention_mask"]
    inputs = {
        "input_ids": input_ids.to(model.device),
        "attention_mask": attention_mask.to(model.device),
    }
    if "attention_mask" not in inspect.signature(model.forward).parameters.keys():
        del inputs["attention_mask"]

    generation_config = GenerationConfig(
        num_return_sequences=num_samples, do_sample=True, max_new_tokens=200
    )
    output_ids = model.generate(**inputs, generation_config=generation_config)
    if model.config.is_encoder_decoder:
        generated_ids = output_ids
    else:
        generated_ids = output_ids[:, input_ids.shape[1] :]

    if num_samples > 1 and isinstance(prompts, list):
        generated_ids = generated_ids.view(input_ids.size(0), num_samples, -1)
    generated_ids = generated_ids.squeeze(0)
    outputs = [tokenizer.decode(ids) for ids in generated_ids.tolist()]
    generation_total_time = time.time() - generation_total_time

    generated_token_num = 0
    for i, output in enumerate(outputs):
        generated_token_num += len(tokenizer.encode(output, add_special_tokens=False))
        print(f"Sample {i + 1}: {output}")
    avg_tps = generated_token_num / generation_total_time
    print(f"Average Tokens Per Second: {avg_tps:.2f}")

    green_count = 0
    for output in outputs:
        tokens = tokenizer.encode(output, add_special_tokens=False)
        for i in range(window_size, len(tokens)):
            context = tokens[i - window_size : i]
            token = tokens[i]
            context_bytes = struct.pack(f">{len(context)}I", *context)
            msg_input = context_bytes + struct.pack(">I", token)
            msg_hash = model.voprf_server.evaluate(msg_input)
            sampling_prob = int.from_bytes(msg_hash, "big") / (1 << 8 * len(msg_hash))
            is_green = sampling_prob < model.gamma
            green_count += is_green
    print(
        f"Green tokens: {green_count} / {generated_token_num} ({green_count / generated_token_num:.2%})"
    )


if __name__ == "__main__":
    main()
