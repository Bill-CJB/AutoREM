# Datasets

The JSON files contain the fixed evaluation inputs and reference objective values used for AutoREM. The validation split is provided for offline adaptation.

| File | Instances | Task |
| --- | ---: | --- |
| `random.json` | 64 | Robust-model reformulation with 2--5 variables |
| `hard.json` | 32 | Reformulation with harder polyhedral uncertainty |
| `large.json` | 32 | Reformulation with 6 or 7 variables |
| `application.json` | 32 | Natural-language formulation followed by robust reformulation |
| `general_p.json` | 32 | Reformulation with non-diagonal uncertainty-loading matrices |
| `coupled_row.json` | 32 | Reformulation with uncertainty shared across rows |
| `external.json` | 32 | Literature-derived robust LP and MILP challenge instances |
| `validation.json` | 64 | Validation-based acceptance during offline adaptation |

## Record format

Each file is a JSON array. Every record includes:

- `id`: an instance identifier, unique within its file;
- `description`: the complete problem input, in LaTeX for reformulation tasks and natural language for the application task;
- `optimal_value`: the stored reference robust objective value.

Reformulation records also include a reference `solution` and structural attributes such as `n_vars`, `n_constraints`, and `uncertainty_type`. The application records retain the nominal benchmark name in `source`; they do not provide reference decision vectors. External records include publication metadata in `source` and a publication-level `source_problem_id`.

Only the problem description is presented to the language models. Objective values, solutions, and source metadata are evaluation data, not prompt content. The application task first formulates the natural-language input as a robust mathematical model and then applies the AutoREM reformulation and coding stages.

The evaluation matches objectives using an absolute error strictly smaller than `1e-4`. Matching objective values is a proxy for reformulation correctness and does not establish mathematical equivalence. Invalid outputs and solver failures count as incorrect. Solver and API behavior may affect repeatability.

## Use and interpretation

Keep the fixed test inputs unchanged when comparing results. Apply the same memory to every test instance without adapting on test feedback. Report every outcome in a run and specify the number of repeated inference runs; those repeats measure inference variability, rather than variability across separately trained memories.

General-P, Coupled-Row, and ExternalRO32 are curated exploratory challenge sets. Their construction or reporting involved inspection of model outputs, so they should not be treated as prospectively fixed, untouched confirmation benchmarks. See [SOURCES.md](SOURCES.md) for construction, provenance, and selection details.
