# VOW: Verifiable and Oblivious Watermark Detection for Large Language Models

This repository contains the source code for the VOW watermarking scheme, a protocol that achieves cryptographic verifiability and privacy-preserving LLM watermarking.
Scripts and raw data for reproducing the main results in the paper are also included.

The proposed scheme, VOW, allows the user to detect watermarks in text without revealing the content of the text to the service provider, thus protecting the user's privacy.
Meanwhile, the detection results returned by the provider come with a proof that can be verified by the user, ensuring the integrity of the watermark detection results.

## Organization

The repository is organized as follows:

- `src/watermark_suite`: Contains the implementation of VOW and baseline methods used for evaluation.
- `voprf-py`: Provides a Python interface for the VOPRF-based watermark scheme, used by our scheme VOW.
- `experiments`: Contains declarative Plans and specialized analysis scripts for evaluating watermark methods.
- `data`: Contains the sampled subset of the C4 dataset used for our evaluation, as well as raw data to plot figures.
- `analysis`: Contains scripts for analysis and visualization of the results.
- `scripts`: Contains a script to download the models and datasets required to reproduce the evaluation results.


## Requirements

- Python >= 3.13
- PyTorch >= 2.7.1
- Transformers >= 4.53.3
- see `pyproject.toml`

We recommend using [uv](https://docs.astral.sh/uv/) to manage the virtual environment and dependencies for this project. To set up the environment, run:

```bash
uv sync
cd voprf-py && maturin develop && cd ..
uv pip install -e .
```

After setting up the environment, `watermark_suite` is installed as a Python package.


## Quick Start

```python
import secrets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from watermark_suite.schemes import VOWAdapter, VOWDetector

# Generate watermarked text
model = AutoModelForCausalLM.from_pretrained("your-model-path")
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("your-model-path")

seed = secrets.token_bytes(32)
adapter = VOWAdapter(
    model, tokenizer, window_size=4, delta=2.5, gamma=0.5, seed=seed
)

prompt = "How many woodchucks would chuck if a woodchuck could chuck wood?"
generated_text = adapter(prompt)  # or a list of str
print(f"Generated: {generated_text}")

# Detect watermark
# 1. the user creates a detector
detector = VOWDetector(tokenizer, window_size=4, gamma=0.5, seed=seed)
# 2. the user gets public key and server interface from the service provider
public_key = adapter.get_public_key()
server_interface = adapter.get_server_interface()
# 3. the user checks watermark
detection_result = detector.detect(generated_text, public_key, server_interface)
# the service provider cannot learn the content of the text
print(detection_result)
```


## Instructions

### Data

We select 500 samples from a subset of C4, `realnewslike`, to construct the dataset for our evaluation experiments.
These samples are saved to a jsonl file under the `data` folder.

Additionally, we stick to one master key (seed) for watermarking purpose to ensure the comparison between different settings is fair.
This master key is also saved under `data` folder.

Other dataset required for replicating the experiments can be downloaded by running the `scripts/download_models_and_datasets.sh` script.

### Experiment Runs

Generation, adaptive forgery, detection, conditional perplexity, and
downstream benchmark evaluation are stage-level Runs described by a YAML
Experiment Plan:

```shell
wmexp check experiments/plans/qwen25-main-and-forgery.yaml
wmexp run experiments/plans/qwen25-main-and-forgery.yaml
wmexp status experiments/plans/qwen25-main-and-forgery.yaml
```

The Plan uses Qwen2.5-7B for generation and adaptive forgery and Qwen2.5-14B
for perplexity. Runs have content-based identities, batch-level crash resume,
SQLite lineage, and immutable Artifact bundles. See
[`docs/experiment-runs.md`](docs/experiment-runs.md) for Plan rules, Workspace
layout, rerun semantics, and legacy-output archival.

The adaptive-forgery stage fixes tokens left to right and queries at most
`k` high-probability candidates through the public blinded VOPRF interface.
The forger receives no watermark key. Its Artifact records per-sample query
counts, query/token ratios, candidate ranks, fallback and cache rates,
local-model perplexity, timing, communication bytes, and optional compact or
full traces. Downstream detection adds the honest-audit comparison, and the
perplexity stage evaluates conditional quality with its own model/tokenizer.
