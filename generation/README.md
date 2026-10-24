# Dataset generation

Run these commands from the repository root after installing the dependencies.
The synthetic generators solve each candidate with RSOME and retain feasible,
bounded instances whose optimum is not entirely at variable bounds. Every
generated record includes `description`, `optimal_value`, and `solution`.

## Synthetic robust linear programs

The generator supports Box, Budgeted, and Polyhedral uncertainty and eight
presentation templates. The template bits select implicit/explicit inner
optimization, scalar/vector presentation, and named/mathematical terminology.
They describe the same numerical problem; they are not AutoREM variants.

```bash
# Random-style data: 64 instances, 16 at each dimension.
python -m generation.synthetic --vars 2 3 4 5 --samples-per-n 16 --seed 42 --output generated/random.json

# Hard-style data: all uncertain rows have Polyhedral uncertainty.
python -m generation.synthetic --vars 2 3 4 5 --samples-per-n 8 --uncertainty-family polyhedral --seed 43 --output generated/hard.json

# Large-style data: 32 instances, 16 at each dimension.
python -m generation.synthetic --vars 6 7 --samples-per-n 16 --seed 44 --output generated/large.json

# Render all eight templates for each accepted numerical instance.
python -m generation.synthetic --vars 2 3 4 5 --samples-per-n 8 --mode all --seed 45 --output generated/all_templates.json
```

The commands above generate new examples with the benchmark construction and
dimensions. The seeds are explicit examples, not a claim to recover the frozen
paper splits byte for byte. Use the supplied `datasets/*.json` files to evaluate
the reported test sets. Use separate seeds for training, validation, and testing.

The training interface is also available as:

```python
from generation.synthetic import batch_generate_json

records = batch_generate_json(
    var_list=[2, 3, 4, 5],
    samples_per_n=8,
    template_list=[[1, 1, 1]],
    seed=42,
    mode="random",
)
```

## Structural extensions

```bash
python -m generation.general_p --vars 2 3 4 5 --samples-per-n 8 --seed 20260726 --output generated/general_p.json
python -m generation.coupled_row --num-samples 32 --var-choices 2 3 4 5 --seed 20260726 --out generated/coupled_row.json
```

`general_p` preserves the uncertainty support and family of every uncertain
objective/constraint row while replacing the coefficient loading with a dense,
full-rank matrix on that support. `coupled_row` makes two existing uncertain rows
share one primitive factor and uncertainty set. All other uncertain rows remain
independent. Both use the compact mathematical presentation template.

## Literature-sourced external set

```bash
python -m generation.external --output generated/external.json
```

This reconstructs the fixed 32 instances, renders their mathematical
descriptions, and recomputes their ground-truth solutions. It needs Gurobi and
a suitable Gurobi license. No LLM calls or model outputs are used by the builder.
The source recipes contain metadata and numerical parameters, not cached
descriptions or answers. Solver ties can produce different optimal solution
vectors with the same optimal objective.

The small source fixtures in `sources/` are:

- Six selection instances from the author-released Goerigk--Khosravi archives:
  [uniform sampler](https://robust-optimization.com/wp-content/uploads/2021/12/mm-b-u.zip),
  [sampler 1](https://robust-optimization.com/wp-content/uploads/2021/12/mm-b-1.zip),
  and [sampler 2](https://robust-optimization.com/wp-content/uploads/2021/12/mm-b-2.zip).
- Three official NETLIB models: [ADLITTLE](https://www.netlib.org/lp/data/adlittle),
  [AFIRO](https://www.netlib.org/lp/data/afiro), and
  [KB2](https://www.netlib.org/lp/data/kb2), rendered as MPS and combined with the
  uncertainty rule in Ben-Tal and Nemirovski (2000).
- Numerical transcriptions from *Robust Optimization* (Ben-Tal, El Ghaoui, and
  Nemirovski, 2009), Li, Ding, and Floudas (2011), and Li, Tang, and Floudas
  (2012), implemented in `_external_models.py`. Full bibliographic details and
  source locators are included in `sources/external_specs.json`.

Upstream benchmark files retain their original provenance and applicable
upstream terms. This repository does not assign them a new license. Full source
papers and benchmark archives are not bundled.

The external set was curated using exploratory runs during its development.
Rebuilding its numerical records does not turn it into an untouched independent
confirmation set; preserve that interpretation when reporting results.

## Application set

`datasets/application.json` contains the 32 curated natural-language application
cases used for evaluation, drawn from IndustryOR and OptiBench with manually
specified uncertainty. No complete automatic script for that curation and
uncertainty annotation is available in this release. The synthetic generators
do not claim to reproduce those application cases.
