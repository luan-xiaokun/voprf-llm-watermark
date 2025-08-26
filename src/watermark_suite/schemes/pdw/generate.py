import argparse
import hashlib
import logging
import os
import pickle
import random
import time
from typing import Any, Callable

import numpy as np
import torch
from bplib.bp import Bn, BpGroup, G1Elem, G2Elem
from petlib.pack import decode, encode
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import crypto, performance_monitor

MAX_TIME_BEFORE_PLANT_ERROR = 300  # seconds
MAX_TIME_BEFORE_GIVE_UP_SAMPLE_VALID_TOKEN = 300  # seconds

STOP_TOKEN = "</s>"

logging.basicConfig(filename="logging.log", encoding="utf-8", level=logging.INFO)


def main(args: argparse.Namespace) -> None:
    generated_text = generate_text(args)
    print(generated_text, file=open("wat.txt", "w"))


def generate_text(args: argparse.Namespace) -> str:
    """Generate text using the specified algorithm and return the generated text."""

    # Manually set pseudorandom seeds for reproducibility.
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        load_in_4bit=args.load_in_4bit,
    )
    logging.info(f"loaded weights in {model.config.torch_dtype}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        max_length=model.config.max_position_embeddings,
        truncation=True,
    )

    (
        generated_text,
        generated_tokens,
        pk,
        params,
        num_tokens,
        num_planted_errors,
    ) = generate_text_asymmetric(
        args.prompt,
        model,
        tokenizer,
        args.sample_type,
        args.message_length,
        args.signature_segment_length,
        args.bit_size,
        args.max_planted_errors,
        args.sk,
        args.pk,
        args.params,
        args.continue_until_stop_token,
    )

    return generated_text


def generate_text_asymmetric(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sample_type: str,
    message_length: int,
    signature_segment_length: int,
    bit_size: int,
    max_planted_errors: int,
    sk_path: str | None,
    pk_path: str | None,
    params_path: str | None,
    continue_until_stop_token: bool = False,
    print_timing: bool = False,
) -> tuple[
    str,
    torch.Tensor,
    bytes,
    tuple[BpGroup, Bn, G1Elem, G2Elem, Callable],
    int,
    int,
]:
    """Generate text using the asymmetric algorithm and return the generated text, tokens, public key, public parameters, token count, and number of planted errors."""

    monitor = performance_monitor.PerformanceMonitor()
    monitor.start_total_timer()

    vocab_size = len(tokenizer)
    inputs = tokenizer.encode(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        padding=False,
        truncation=False,
    ).to(model.device)

    # Save initial_inputs_len to truncate at the end to exclude the prompt from the generated_text output.
    initial_inputs_len = inputs.size(1)

    attn = torch.ones_like(inputs)
    past: torch.Tensor | None = None

    sk: list = []
    pk: bytes = b""
    params: tuple = ()

    # Load or generate the secret key, public key, and parameters. We will use the secret key and public parameters to generate the watermark. The public key and parameters will be used later for detection.
    if (sk_path and pk_path and params_path) and (
        os.path.exists(sk_path)
        and os.path.exists(pk_path)
        and os.path.exists(params_path)
    ):
        logging.info(
            f"found existing sk, pk, and params: loading sk from {sk_path}, pk from {pk_path}, and params from {params_path}"
        )
        with open(sk_path, "rb") as f:
            sk = decode(pickle.load(f))
        with open(pk_path, "rb") as g:
            pk = decode(pickle.load(g))
        with open(params_path, "rb") as h:
            G = decode(pickle.load(h))
            params = (G, G.order(), G.gen1(), G.gen2(), G.pair)
    else:
        # If paths are't specified or the files don't exist at the specified paths, generate a new set of secret key, public key, and parameters.
        logging.info("generating new sk, pk, and params")
        sk, pk, params = crypto.bls_generate_openssl()
        (G, o, g1, g2, e) = params

        # Save to pickles if the paths are specified. (Specified when using CLI, unspecified for unit tests and benchmarking.)
        if sk_path:
            logging.info(f"saving sk to {sk_path}")
            with open(sk_path, "wb") as f:
                pickle.dump(encode(sk), f)
        if pk_path:
            logging.info(f"saving pk to {pk_path}")
            with open(pk_path, "wb") as g:
                pickle.dump(encode(pk), g)
        if params_path:
            logging.info(f"saving params to {params_path}")
            with open(params_path, "wb") as h:
                pickle.dump(encode(G), h)

    counter = 0
    generated_text = ""

    if continue_until_stop_token:
        # Embed first message-signature pair (without stop token), then continue until we get a stop token.
        embedded_first_message_signature_pair = False
        while True:
            (
                message_signature_pair,
                inputs,
                past,
                attn,
                counter,
                planted_errors,
            ) = generate_message_signature_pair(
                message_length,
                signature_segment_length,
                bit_size,
                max_planted_errors,
                sk,
                params,
                model,
                tokenizer,
                vocab_size,
                sample_type,
                inputs,
                past,
                attn,
                counter,
                monitor,
                embedded_first_message_signature_pair,
            )
            generated_text += message_signature_pair
            embedded_first_message_signature_pair = True
            if STOP_TOKEN in generated_text:
                generated_text = generated_text[: generated_text.index(STOP_TOKEN)]
                break
    else:
        # Just embed one message-signature pair.
        (
            message_signature_pair,
            inputs,
            past,
            attn,
            counter,
            planted_errors,
        ) = generate_message_signature_pair(
            message_length,
            signature_segment_length,
            bit_size,
            max_planted_errors,
            sk,
            params,
            model,
            tokenizer,
            vocab_size,
            sample_type,
            inputs,
            past,
            attn,
            counter,
            monitor,
            False,
        )
        generated_text += message_signature_pair

    monitor.stop_total_timer()
    generated_tokens = inputs[:, initial_inputs_len:]
    final_accepted_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
    num_accepted_tokens = len(final_accepted_tokens)

    if print_timing:
        monitor.calculate_and_print_results(num_generated_tokens=num_accepted_tokens)

    return (
        generated_text,
        generated_tokens,
        pk,
        params,
        counter,
        planted_errors,
    )


def generate_message_signature_pair(
    message_length: int,
    signature_segment_length: int,
    bit_size: int,
    max_planted_errors: int,
    sk: list,
    params: tuple,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    vocab_size: int,
    sample_type: str,
    inputs: torch.Tensor,
    past: torch.Tensor | None,
    attn: torch.Tensor,
    counter: int,
    monitor: performance_monitor.PerformanceMonitor,
    embedded_first_message_signature_pair: bool,
) -> tuple[
    str,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    int,
    int,
]:
    """
    Generate one message-signature pair.

    Returns:
        - message_signature_pair (str): the message and signature concatenated together
        - inputs (torch.Tensor): the updated inputs after sampling enough characters
        - past (torch.Tensor | None): the updated past after sampling enough characters
        - attn (torch.Tensor): the updated attn after sampling enough characters
        - counter (int): the updated counter after sampling enough characters
        - planted_errors (int): the number of errors planted in the signature
    """

    message_signature_pair = ""

    (
        message,
        start_of_signature_segment,
        inputs,
        past,
        attn,
        counter,
    ) = sample_n_characters(
        message_length,
        "",  # message starts off empty
        model,
        tokenizer,
        vocab_size,
        sample_type,
        inputs,
        past,
        attn,
        counter,
        monitor,
        embedded_first_message_signature_pair,
    )
    message_signature_pair += message

    if STOP_TOKEN in message_signature_pair:
        return message_signature_pair, inputs, past, attn, counter, 0

    # Get the message and signature.
    message_hash = hashlib.sha256(bytes(message, "utf-8")).digest()
    signature = crypto.sign_and_encode_openssl(
        sk, message_hash, params, max_planted_errors
    )

    logging.info(f"signature\n{signature}")

    prev_sig_bits = ""
    planted_errors = 0
    signature_codeword_length = crypto.get_signature_codeword_length(
        max_planted_errors, bit_size
    )

    # Embed the signature
    for i in tqdm(range(signature_codeword_length // bit_size)):
        if STOP_TOKEN in message_signature_pair:
            return message_signature_pair, inputs, past, attn, counter, planted_errors

        # Save a copy of inputs, past, and attn before sampling each signature segment
        # in case we need to retry the sample for a signature segment.
        inputs_before_signature_sampling = inputs
        past_before_signature_sampling = past
        attn_before_signature_sampling = attn

        start_of_signature_segment_before_signature_sampling = (
            start_of_signature_segment
        )
        is_signature_segment_valid = False
        signature_bit, signature = (
            signature[:bit_size],
            signature[bit_size:],
        )
        counter_before_signature_segment_sampling = counter

        initial_time = time.time()
        best_match_value = -1
        best_match = ""
        best_inputs = torch.Tensor()
        best_past = None
        best_attn = torch.Tensor()
        best_unkeyed_hash = None
        best_next_signature_segment = ""

        while not is_signature_segment_valid:
            # Each time we retry the signature segment sample, we need to reset inputs, past, attn, and counter.
            inputs = inputs_before_signature_sampling
            past = past_before_signature_sampling
            attn = attn_before_signature_sampling
            counter = counter_before_signature_segment_sampling

            (
                signature_segment,
                next_signature_segment,
                inputs,
                past,
                attn,
                counter,
            ) = sample_n_characters(
                signature_segment_length,
                start_of_signature_segment_before_signature_sampling,  # signature_segment starts with overflow from previous signature_segment (or overflow from message for the first signature_segment)
                model,
                tokenizer,
                vocab_size,
                sample_type,
                inputs,
                past,
                attn,
                counter,
                monitor,
                embedded_first_message_signature_pair,
            )

            # Check if the signature_segment_length many tokens hash to the signature.
            unkeyed_hash = crypto.unkeyed_hash_to_bits(
                bytes(message, "utf-8")
                + bytes(prev_sig_bits, "utf-8")
                + bytes(signature_segment, "utf-8"),
                bit_size,
            )

            is_signature_segment_valid = int(unkeyed_hash) == int(signature_bit)

            use_planted_errors = max_planted_errors > 0
            if use_planted_errors and not is_signature_segment_valid:

                curr_match_value = sum(
                    [
                        1 if unkeyed_hash[i] == signature_bit[i] else 0
                        for i in range(len(unkeyed_hash))
                    ]
                )
                if curr_match_value > best_match_value:
                    best_match_value = curr_match_value
                    best_match = signature_segment
                    best_inputs = inputs
                    best_past = past
                    best_attn = attn
                    best_unkeyed_hash = unkeyed_hash
                    best_next_signature_segment = next_signature_segment

                if (
                    (time.time() - initial_time) >= MAX_TIME_BEFORE_PLANT_ERROR
                    and planted_errors < max_planted_errors
                ):
                    signature_segment = best_match
                    inputs = best_inputs
                    past = best_past
                    attn = best_attn
                    unkeyed_hash = best_unkeyed_hash
                    next_signature_segment = best_next_signature_segment

                    planted_errors += bit_size - best_match_value
                    logging.info(
                        f"planting {bit_size-best_match_value} errors, best_match: {best_match}, total errors: {planted_errors}"
                    )
                    break
                elif (
                    (time.time() - initial_time) >= MAX_TIME_BEFORE_PLANT_ERROR
                    and planted_errors >= max_planted_errors
                ):
                    # We've already planted the maximum number of errors, so we can't plant any more. Throw an exception.
                    logging.info(
                        f"tried to plant another error but already at max_planted_errors: {max_planted_errors}"
                    )
                    raise ValueError(
                        f"tried to plant another error but already at max_planted_errors: {max_planted_errors}"
                    )
            elif (time.time() - initial_time) >= MAX_TIME_BEFORE_PLANT_ERROR:
                logging.info(
                    f"signature segment {signature_segment} took too long to hash to {signature_bit}, raising exception"
                )
                raise ValueError(
                    f"signature segment {signature_segment} took too long to hash to {signature_bit}"
                )

        prev_sig_bits += unkeyed_hash
        message_signature_pair += signature_segment
        start_of_signature_segment = next_signature_segment

    logging.info(f"error signature\n{prev_sig_bits}")

    return message_signature_pair, inputs, past, attn, counter, planted_errors


def sample_n_characters(
    n: int,
    start_of_n_characters: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    vocab_size: int,
    sample_type: str,
    inputs: torch.Tensor,
    past: torch.Tensor | None,
    attn: torch.Tensor,
    counter: int,
    monitor: performance_monitor.PerformanceMonitor,
    embedded_first_message_signature_pair: bool,
) -> tuple[
    str,
    str,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    int,
]:
    """
    Sample n characters using the provided inputs.

    Returns:
        - n_character_str (str): the n characters sampled
        - overflow_str (str): the characters that were sampled after n_character_str
        - inputs (torch.Tensor): the updated inputs after sampling enough tokens
        - past (torch.Tensor | None): the updated past after sampling enough tokens
        - attn (torch.Tensor): the updated attn after sampling enough tokens
        - counter (int): the updated counter after sampling enough tokens
    """

    n_character_str = start_of_n_characters
    overflow_str = ""

    is_prefill_step = past is None

    while len(n_character_str) < n:
        # Save prev_inputs to use in the workaround for models which don't preserve spaces when decoding token-by-token.
        prev_inputs = inputs

        monitor.record_step_start()

        token, inputs, past, attn = sample_token(
            model,
            tokenizer,
            inputs,
            past,
            attn,
            vocab_size,
            sample_type,
            embedded_first_message_signature_pair,
        )

        monitor.record_step_end(is_prefill=is_prefill_step)
        is_prefill_step = False

        counter += 1
        token_str = decode_token(
            token,
            prev_inputs,
            inputs,
            tokenizer,
        )

        n_character_str += token_str

        if len(n_character_str) > n:
            overflow_str = n_character_str[n:]
            n_character_str = n_character_str[:n]

        if token_str == STOP_TOKEN:
            break

    return (
        n_character_str,
        overflow_str,
        inputs,
        past,
        attn,
        counter,
    )


def decode_token(
    token: torch.Tensor,
    prev_inputs: torch.Tensor,
    inputs: torch.Tensor,
    tokenizer: AutoTokenizer,
) -> str | Any:
    """Decode a single token and return the decoded token string."""

    token_str = tokenizer.decode(token.squeeze().detach().cpu())
    prev_inputs_str = tokenizer.decode(prev_inputs.squeeze().detach().cpu())
    new_inputs_str = tokenizer.decode(inputs.squeeze().detach().cpu())
    token_str = new_inputs_str[len(prev_inputs_str) :]
    return token_str


def sample_token(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: torch.Tensor,
    past: torch.Tensor | None,
    attn: torch.Tensor,
    vocab_size: int,
    sample_type: str,
    embedded_first_message_signature_pair: bool = False,
    top_p: float = 0.9,
    temperature: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    Sample a token token along with updated inputs, past, and attn.

    If we haven't embedded the first message-signature pair and get a stop token, we try again until we get a non-stop token.
    If we're on a subsequent message-signature pair, we exit if we get a stop token.
    """
    sampled_valid_token = False
    num_sample_attempts = 0

    initial_time = time.time()
    while not sampled_valid_token:
        if (time.time() - initial_time) >= MAX_TIME_BEFORE_PLANT_ERROR:
            logging.info(
                f"sample_token took too long to sample a valid token, raising exception"
            )
            raise ValueError(f"sample_token took too long to sample a valid token")

        with torch.no_grad():
            if past:
                output = model(
                    inputs[:, -1:],
                    past_key_values=past,
                    attention_mask=attn,
                )
            else:
                output = model(inputs)

            logits = output.logits[:, -1, :vocab_size]

            if sample_type == "argmax":
                probs = torch.nn.functional.softmax(logits, dim=-1)
                token = torch.argmax(probs, dim=-1)
            elif sample_type == "multinomial":
                probs = torch.nn.functional.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            elif sample_type == "nucleus":
                sorted_logits, sorted_indices = torch.sort(
                    logits / temperature, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[..., indices_to_remove] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            else:
                raise ValueError(f"sample_type {sample_type} not supported")

            num_sample_attempts += 1
            token_str = tokenizer.decode(token.squeeze().detach().cpu())
            if not embedded_first_message_signature_pair and STOP_TOKEN in token_str:
                logging.info(f"sampled stop token: {token_str}, retrying")
                sampled_valid_token = False
                continue
            else:
                sampled_valid_token = True

            token = torch.tensor([[token]], device=inputs.device)
            inputs = torch.cat([inputs, token], dim=-1)
            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return token, inputs, past, attn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate text")
    parser.add_argument(
        "--prompt", default="There once was a", type=str, help="the generation prompt"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B",
        type=str,
        help="the id of the Hugging Face model to use",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        help="whether to load the model with 4-bit quantization",
    )
    parser.add_argument(
        "--num-tokens", default=80, type=int, help="the number of generated tokens"
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="a seed for the PyTorch random number generator",
    )
    parser.add_argument(
        "--sample-type",
        default="multinomial",
        type=str,
        help="the type of sampling to use at generation time, one of: 'argmax', 'multinomial', 'nucleus'",
        choices=("argmax", "multinomial", "nucleus"),
    )

    # The following arguments are used in asymmetric watermarking only: sk, pk, params, signature_segment_length, bit_size, max_planted_errors, continue_until_stop_token
    parser.add_argument(
        "--sk",
        default="sk.pickle",
        type=str,
        help="the path for the secret generation key pickle file; if a file already exists at this path, it will be loaded and reused",
    )
    parser.add_argument(
        "--pk",
        default="pk.pickle",
        type=str,
        help="the output path for the public detection key pickle file",
    )
    parser.add_argument(
        "--params",
        default="params.pickle",
        type=str,
        help="the path for the params pickle file; if a file already exists at this path, it will be loaded and reused",
    )
    parser.add_argument(
        "--message-length",
        default=crypto.DEFAULT_MESSAGE_LENGTH,
        type=int,
        help="the length of the message in characters. The default value is signature-segment-length // bit-size",
    )
    parser.add_argument(
        "--signature-segment-length",
        default=crypto.DEFAULT_SIGNATURE_SEGMENT_LENGTH,
        type=int,
        help="the length of each signature segment in characters",
    )
    parser.add_argument(
        "--bit-size",
        default=crypto.DEFAULT_BIT_SIZE,
        type=int,
        help="the number of signature bits in each segment",
    )
    parser.add_argument(
        "--max-planted-errors",
        default=crypto.DEFAULT_MAX_PLANTED_ERRORS,
        type=int,
        help="the number of error correcting symbols to use",
    )
    parser.add_argument(
        "--continue-until-stop-token",
        action=argparse.BooleanOptionalAction,
        help="flag to keep sampling until a stop token after the first message-signature pair is embedded",
    )

    main(parser.parse_args())
