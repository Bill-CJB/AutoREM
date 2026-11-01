# AutoREM

Code and datasets for **Automated Reformulation of Robust Optimization via Memory-Augmented Large Language Models**, NeurIPS 2026.

AutoREM learns a compact reformulation experience memory through offline adaptation. A reformulator uses this memory to produce a deterministic robust counterpart; a coder converts the counterpart into executable Gurobi code. Reflection, step-level double-checking, and validation-based acceptance control memory updates during training. Evaluation uses a frozen memory.

## Contents

```text
autorem/       AutoREM memory, prompts, adaptation, execution, and command-line tools
generation/    Random robust optimization problem generation for offline training
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

The dependencies include NumPy, SciPy, RSOME, Gurobi, and tqdm. The LLM client uses the Python standard library and the DeepSeek chat-completions API. Dataset and memory paths in the commands below are relative to the repository root.

## API configuration

Create a DeepSeek API key and set it in your shell before running training or evaluation. One key serves all roles. For Bash:

```bash
export AUTOREM_API_KEY="your-api-key"
export AUTOREM_BASE_URL="https://api.deepseek.com/v1"
export AUTOREM_MODEL="deepseek-v4-flash"
export AUTOREM_CODER_MODEL="deepseek-v4-pro"
```

For PowerShell:

```powershell
$env:AUTOREM_API_KEY = "your-api-key"
$env:AUTOREM_BASE_URL = "https://api.deepseek.com/v1"
$env:AUTOREM_MODEL = "deepseek-v4-flash"
$env:AUTOREM_CODER_MODEL = "deepseek-v4-pro"
```

All training and evaluation stages use `https://api.deepseek.com/v1`:

| Role | Default model | Additional request fields |
| --- | --- | --- |
| Reformulator, reflector, and application formulator | `deepseek-v4-flash` | 32,768 output-token limit, JSON-object format |
| Coder | `deepseek-v4-pro` | No explicit token-limit or response-format fields |

All roles use temperature 0, the same JSON-only system instruction, and the API's default reasoning setting. These are the Flash and Pro model roles used in the main experiments. No provider-specific thinking fields are added to the requests.

The experiments used DeepSeek-V4-Flash as the base LLM and DeepSeek-V4-Pro as the fixed coder.

The coder shares `AUTOREM_API_KEY` and the base URL by default. Set `AUTOREM_CODER_API_KEY` only if you want to use a separate DeepSeek key for coding. Keep credentials in your environment; do not insert them into source files. See [.env.example](.env.example) for optional model, token-limit, and timeout settings. The example file is documentation and is not loaded automatically.

For a small configuration check, run `python -m autorem evaluate --dataset random --workers 1 --limit 1 --output outputs/smoke`. HTTP authentication or configuration errors stop the run with an error message. Model availability, API behavior, and stochastic inference can affect results. API calls incur charges according to DeepSeek's pricing.

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

The release contains 256 test instances: Random64, Hard32, Large32, Application32, General-P32, Coupled-Row32, and ExternalRO32. Large32 contains sixteen six-variable and sixteen seven-variable problems, matching the fixed inputs in the main experiments. Validation64 is included for training. See [datasets/README.md](datasets/README.md) for the schema and [datasets/SOURCES.md](datasets/SOURCES.md) for source attribution and construction details.

The test and validation sets are distributed as fixed JSON files. The only generation code supplied is the random training-problem generator used during offline adaptation; see [generation/README.md](generation/README.md). Training generates samples as needed without saving a fixed training set first.

General-P, Coupled-Row, and ExternalRO32 are curated exploratory challenge sets whose construction or reporting involved inspection of model outputs. Evaluate all released cases and report all repeated runs; these sets should not be treated as untouched confirmation benchmarks.

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
