# we need a specification for saving generated outputs by various watermarking
# methods and the normal generation process without watermark
# the specification includes details of the output format, file naming conventions,
# and the organization of the output files.

# 1. Output Format
# The output format for all generated files should be JSON Lines (JSONL), where
# each line is a valid JSON object.
# This format is chosen for its simplicity and ease of use with large datasets.
# A metadata json file is created for each generated output file.
# The metadata json file is suffixed by ".meta.json" instead of ".jsonl",
# containing the parameters used for generation and the file name of the jsonl
# file, and additionally a timestamp.

# - We would need a script to automatically convert the files we have obtained
#   into the specified output format. This needs to be done with caution.

# 2. File Naming Conventions
# Generated files should be named in a way that fully describes the setting
# used for generation. This includes:
# - watermarking method, e.g., vow, selfhash, rdf, pdw, or no-watermark
# - watermark parameters, e.g., delta, gamma, window_size, etc.
# - dataset name, e.g., c4, eli5, etc
# - language model, e.g., qwen2.5-3b, qwen2.5-3b-instruct
# - decoding strategy, e.g., multinomial, top-k, default (for rdf)
# - modification, e.g., synonym replaced, paraphrased
# these components should be concatenated into a single string, separated by
# underscores (i.e., each part should not contain underscores)
# The order is as follows:
# - <watermarking-method>
# - <language-model>
# - <dataset>
# - <decoding-strategy>
# - <watermark-parameters>
# - <modification>

# 3. Organization of Output Files
# Output files should be organized into directories based on some criteria.
# The most high-level criterion is the purpose of the generation.
# - output/generation, for the main generation outputs
# - output/human_eval and output/gsm8k, for downstream task evaluation
# - output/robustness, for robustness evaluation
# - output/forgery, for learning attack simulation
import json
import os
import re
import shutil
import tempfile
import warnings
from collections.abc import Generator
from pathlib import Path

WATERMARKING_METHODS = [
    "vow",
    "lefthash",
    "selfhash",
    "rdf",
    "no-watermark",
    "pdw",
    "upv",
]
DATASET_NAMES = ["c4", "eli5"]
MODIFICATIONS = ["synonym-substituted", "paraphrased"]


def read_json(file_name: str) -> dict:
    with open(file_name, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(file_name: str, data: dict, indent: int | None = None) -> None:
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)


def read_jsonlines(file_name: str) -> Generator[dict, None, None]:
    """Read a JSON Lines file and yield each JSON object."""
    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def write_jsonlines(
    file_name: str, data: dict | list[dict], mode: str = "w", indent: int | None = None
) -> None:
    """Append JSON objects to a JSON Lines file."""
    if isinstance(data, dict):
        data = [data]
    with open(file_name, mode, encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, indent=indent, ensure_ascii=False) + "\n")


def get_decoding_strategy(
    do_sample: bool | None,
    top_k: int | None,
) -> str:
    if not do_sample:
        if top_k:
            warnings.warn("Ignoring top_k parameter when do_sample is False")
        return "greedy"

    if top_k is None:
        return "multinomial"

    return f"top-{top_k}"


def parse_decoding_strategy(decoding: str) -> dict:
    if decoding in ["greedy", "default"]:
        return {
            "do_sample": False,
            "top_k": None,
            "top_p": None,
            "temperature": None,
        }
    if decoding == "multinomail":
        return {
            "do_sample": True,
            "top_k": None,
            "top_p": None,
            "temperature": 0.7,
        }
    if decoding.startswith("top-"):
        top_k = int(decoding.split("-")[1])
        return {
            "do_sample": True,
            "top_k": top_k,
            "top_p": 0.9,
            "temperature": 0.7,
        }
    raise ValueError(f"Unknown decoding strategy: {decoding}")


def watermark_parameter_dict_to_string(method: str, params: dict) -> str:
    if method == "no-watermark":
        return ""
    if method == "vow":
        window_size = params.get("window_size")
        delta = params.get("delta")
        gamma = params.get("gamma")
        if window_size is None or gamma is None or delta is None:
            raise ValueError("VOW missing watermark parameters")
        delta = f"{delta:.1f}" if delta.is_integer() else str(delta)
        gamma = f"{gamma:.1f}" if gamma.is_integer() else str(gamma)
        return f"w{window_size}_d{delta}_g{gamma}"
    if method in ["lefthash", "selfhash"]:
        delta = params.get("delta")
        gamma = params.get("gamma")
        if gamma is None or delta is None:
            raise ValueError("KGW missing watermark parameters")
        delta = f"{delta:.1f}" if delta.is_integer() else str(delta)
        gamma = f"{gamma:.1f}" if gamma.is_integer() else str(gamma)
        return f"d{delta}_g{gamma}"
    if method == "rdf":
        # we use the same default length as in RDF's implementation
        # this parameter is the `n` in the paper
        length = params.get("length", 256)
        # we use the same default seed as in RDF's implementation
        seed = params.get("seed", 42)
        # we use the default cpu device as in RDF's implementation
        watermark_device = params.get("watermark_device", "cpu")
        if watermark_device == "cuda":
            watermark_device = "gpu"
        return f"l{length}_s{seed}_{watermark_device}"
    if method == "pdw":
        return ""
    if method == "upv":
        return "18bits_5layers"
    raise ValueError(f"Unknown watermarking method: {method}")


def get_file_name(
    method: str,
    model: str,
    dataset: str,
    watermark_params: dict,
    do_sample: bool | None = None,
    top_k: int | None = None,
    modification: str | list[str] | None = None,
) -> str:
    assert method in WATERMARKING_METHODS, f"Invalid watermarking method: {method}"
    assert dataset in DATASET_NAMES, f"Invalid dataset name: {dataset}"

    model = str(model).split("/")[-1]  # Use the last part of the model path
    decoding = get_decoding_strategy(do_sample, top_k)
    if method == "rdf":
        decoding = "default"

    params = watermark_parameter_dict_to_string(method, watermark_params)

    if modification is None:
        modification = []
    if not isinstance(modification, list):
        modification = [modification]
    for mod in modification:
        assert mod in MODIFICATIONS, f"Invalid modification: {mod}"

    file_name = f"{method}_{model}_{dataset}_{decoding}"

    if params:
        file_name += f"_{params}"
    if modification:
        modification = "_".join(sorted(modification))
        file_name += f"_{modification}"

    file_name += ".jsonl"

    return file_name


def parse_file_name(file_name: str) -> dict:
    components = file_name.rstrip(".jsonl").split("_")
    if len(components) < 4:
        raise ValueError(f"Invalid file name format: {file_name}")

    method = components[0]
    model = components[1]
    dataset = components[2]
    decoding = components[3]

    modification = []
    for comp in components[::-1]:
        if comp in MODIFICATIONS:
            modification.append(comp)
    modification = modification[::-1]

    params_size = len(components) - len(modification) - 4
    params = components[4 : 4 + params_size]
    param_dict = {}
    for param in params:
        if match := re.match(r"w(.*)", param):
            param_dict["window_size"] = int(match.group(1))
        elif match := re.match(r"d(.*)", param):
            param_dict["delta"] = float(match.group(1))
        elif match := re.match(r"g(.*)", param):
            param_dict["gamma"] = float(match.group(1))
        elif match := re.match(r"l(.*)", param):
            param_dict["length"] = int(match.group(1))
        elif match := re.match(r"s(.*)", param):
            param_dict["seed"] = int(match.group(1))
        elif match := re.match(r"cpu|gpu", param):
            param_dict["watermark_device"] = match.group(0)
        else:
            raise ValueError(f"Invalid parameter: {param}")

    return {
        "method": method,
        "model": model,
        "dataset": dataset,
        **parse_decoding_strategy(decoding),
        **param_dict,
    }


def merge_and_write_jsonl(
    filepath: str | Path,
    original_samples: list[dict],
    results_to_add: dict[str, list],
) -> None:
    """
    Safely merges new results into a list of samples and writes them back to a JSONL file.

    This function first validates that all lists have the same length, then merges the
    data in memory, and finally uses an atomic write pattern to save the result.

    Args:
        filepath (Union[str, Path]): The path to the original JSONL file.
        original_samples (List[Dict]): The list of original JSON objects.
        results_to_add (Dict[str, List[Any]]): A dictionary where keys are the new
            field names and values are the lists of results to add.
    """
    num_samples = len(original_samples)
    for field_name, result_list in results_to_add.items():
        if len(result_list) != num_samples:
            raise ValueError(
                f"Data length mismatch. Original samples have {num_samples} items, "
                f"but the result list for '{field_name}' has {len(result_list)} items."
            )

    for i, sample in enumerate(original_samples):
        for field_name, result_list in results_to_add.items():
            sample[field_name] = result_list[i]

    source_path = Path(filepath)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=source_path.parent,
        prefix=f".{source_path.name}_",
        suffix=".tmp",
    )

    try:
        for sample in original_samples:
            temp_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

        temp_file.close()

        shutil.move(temp_file.name, source_path)
        print(f"Successfully merged results and updated {source_path}")

    except Exception as e:
        print(f"An error occurred during file writing: {e}")
        print("Operation failed. The original file has not been changed.")
    finally:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
