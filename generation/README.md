# Random training data

This module generates fresh robust linear programs for AutoREM's offline
adaptation. Each instance has Box, Budgeted, or Polyhedral uncertainty and
exactly one deterministic row among its objective and constraints. RSOME solves
the robust model to provide a ground-truth objective and solution. The generator
retains feasible, bounded instances whose optimum is not entirely at variable
bounds.

Run from the repository root after installing the dependencies:

```bash
python -m generation.synthetic --vars 2 3 4 5 --samples-per-n 8 --seed 42 --output generated/train.json
```

The command produces 32 fresh training records. Each record includes
`description`, `optimal_value`, and `solution`. Its presentation is sampled from
eight templates, whose three bits select implicit/explicit inner optimization,
scalar/vector notation, and named/mathematical terminology. These templates
describe the same numerical problem. Use `--templates 111` to generate training
descriptions in one chosen presentation.

AutoREM also calls the generator directly during adaptation:

```python
from generation.synthetic import batch_generate_json

records = batch_generate_json(
    var_list=[2, 3, 4, 5],
    samples_per_n=8,
    template_list=[
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ],
    seed=42,
    mode="random",
)
```

Use a seed when you need repeatable training generation. With `seed=None`, the
function uses the caller's current random state, as in on-the-fly adaptation.
The evaluation datasets are supplied separately in `datasets/`.
