# AutoREM

Code and datasets for **Automated Reformulation of Robust Optimization via Memory-Augmented Large Language Models**, NeurIPS 2026.

AutoREM learns a compact reformulation experience memory through offline adaptation. A reformulator uses this memory to produce a deterministic robust counterpart; a coder converts the counterpart into executable Gurobi code. Reflection, step-level double-checking, and validation-based acceptance control memory updates during training. Evaluation uses a frozen memory.

## Contents

```text
autorem/       AutoREM memory, prompts, adaptation, execution, and command-line tools
generation/    Dataset generators and required numerical source fixtures
datasets/      Seven fixed test sets and the offline validation set
memory/        Final learned AutoREM memory (eight experiences)
```

## Installation

Use Python 3.10 or newer and a Gurobi installation with a license that supports the problem sizes being solved. From the repository root:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e .
```

The dependencies include NumPy, SciPy, RSOME, Gurobi, and tqdm. The LLM client uses the Python standard library and an OpenAI-compatible chat-completions endpoint. Dataset and memory paths in the commands below are relative to the repository root.

## API configuration

Set API credentials in your environment. For Bash:

```bash
export AUTOREM_API_KEY="your-api-key"
export AUTOREM_CODER_API_KEY="your-coder-api-key"
export AUTOREM_MODEL="deepseek-v4-flash"
export AUTOREM_CODER_MODEL="deepseek-v4-pro"
```

For PowerShell:

```powershell
$env:AUTOREM_API_KEY = "your-api-key"
$env:AUTOREM_CODER_API_KEY = "your-coder-api-key"
$env:AUTOREM_MODEL = "deepseek-v4-flash"
$env:AUTOREM_CODER_MODEL = "deepseek-v4-pro"
```

The base model serves the reformulator, reflector, and application formulator. The coder uses its own model and credential. Default endpoints and request settings reproduce the role settings in the evaluation and training scripts:

| Role/task | Default endpoint | Additional request fields |
| --- | --- | --- |
| Mathematical-model evaluation, base model | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Thinking enabled, high reasoning effort, 32,768 output-token limit, JSON-object format |
| Offline training and application evaluation, base model | `https://api.deepseek.com/v1` | 32,768 output-token limit, JSON-object format |
| Coder in all tasks | `https://api.deepseek.com/v1` | No explicit thinking, token-limit, or response-format fields |

All roles use temperature 0 and the same JSON-only system instruction. Set `AUTOREM_API_KEY` for the base endpoint used by your command, and `AUTOREM_CODER_API_KEY` for DeepSeek. When changing from mathematical-model evaluation to training or application evaluation, update the base credential accordingly. `AUTOREM_BASE_URL` and `AUTOREM_CODER_BASE_URL` override endpoints. A base credential is reused for the coder only if both roles explicitly target the same URL.

See [.env.example](.env.example) for API field overrides. The example file is documentation and is not loaded automatically. An alternative provider must support the configured request fields. Model availability, provider behavior, and stochastic inference can affect results. API calls incur charges according to your provider.

## Evaluate the supplied memory

```bash
python -m autorem evaluate --dataset random --output outputs/random
python -m autorem evaluate --dataset hard --output outputs/hard
python -m autorem evaluate --dataset large --output outputs/large
python -m autorem evaluate --dataset application --output outputs/application
python -m autorem evaluate --dataset general_p --output outputs/general_p
python -m autorem evaluate --dataset coupled_row --output outputs/coupled_row
python -m autorem evaluate --dataset external --output outputs/external
```

Evaluation defaults to `memory/autorem.json` and four worker threads. Use `--workers 1 --limit 1` for a small smoke run. `--memory PATH` loads a trained memory; `--data PATH` supplies a JSON dataset using the selected dataset's pipeline. The application pipeline first formulates the natural-language input, then reformulates and codes the robust counterpart. Other datasets begin with a mathematical model.

Only the problem description is passed to the language models. Reference objective values and solutions are used for scoring. A sample is correct when execution produces a finite objective value with absolute error strictly smaller than `1e-4`. Invalid outputs and solver failures count as incorrect. API transport or configuration failures abort the run. Outputs include per-sample model responses and solver outcomes, plus aggregate accuracy and token counts in `metrics.json`. Token usage is the usage returned for the completion selected by each stage; retries can incur additional usage.

Generated solver code runs in a temporary directory with a timeout and common API credential environment variables removed. This subprocess is not a security sandbox. Use a suitably isolated execution environment for untrusted inputs or generated code.

## Train memory

```bash
python -m autorem train --validation datasets/validation.json --output outputs/training
```

Training starts with empty memory and generates fresh robust optimization problems. The final procedure uses ten epochs, eight steps per epoch, one training problem per step, and up to three reflection iterations. Each step-level double-check uses up to four validation samples. Candidate updates must pass these checks; epoch acceptance uses validation accuracy, including ties. The validation set is required.

Each run writes its logs and checkpoints to a timestamped directory. `best_rem.json` contains the memory selected by validation. To resume an epoch checkpoint:

```bash
python -m autorem train --resume outputs/training/RUN/trained_rem_epoch5.json --output outputs/training
```

Use the actual epoch-checkpoint filename produced by your run. Training makes many LLM and solver calls; inference repeats with one frozen memory do not measure variability across independently trained memories.

## Data and generation

The release contains 256 test instances: Random64, Hard32, Large32, Application32, General-P32, Coupled-Row32, and ExternalRO32. Validation64 is included for training. See [datasets/README.md](datasets/README.md) for the schema and [datasets/SOURCES.md](datasets/SOURCES.md) for source attribution and construction details.

Generation commands and reproducibility limits are documented in [generation/README.md](generation/README.md). General-P, Coupled-Row, and ExternalRO32 are curated exploratory challenge sets whose construction or reporting involved inspection of model outputs. Evaluate all released cases and report all repeated runs; these sets should not be treated as untouched confirmation benchmarks. Exact regeneration of the curated natural-language Application32 cases is not claimed.

## Citation

```bibtex
@inproceedings{chen2026autorem,
  title     = {Automated Reformulation of Robust Optimization via Memory-Augmented Large Language Models},
  author    = {Chen, Jinbiao and Jin, Shuang and Zhang, Guoyun and Zhang, Junyu and Wang, Guanyi and Qin, Hanzhang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2026)},
  year      = {2026},
  url       = {https://github.com/Bill-CJB/AutoREM}
}
```
