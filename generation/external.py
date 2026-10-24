"""Rebuild the 32 literature-sourced external evaluation instances.

The recipes contain source metadata and fixed parameters. Descriptions and
ground-truth answers are constructed from the mathematical models at runtime.
"""

from __future__ import annotations

import argparse
import copy
import json
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory


SOURCE_DIR = Path(__file__).resolve().parent / "sources"


def _solve_netlib(source: Path, epsilon: Decimal, delta: Decimal):
    import gurobipy as gp
    from gurobipy import GRB

    from . import _netlib

    with TemporaryDirectory(prefix="autorem_netlib_") as folder:
        counterpart = Path(folder) / "counterpart.lp"
        _netlib.build_counterpart(source, epsilon, delta, counterpart)
        model = gp.read(str(counterpart))
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.NumericFocus = 2
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"NETLIB solve failed with status {model.Status}")
        _, _, columns, _, _, _, _ = _netlib.parse(source)
        solution = [model.getVarByName(f"x{i}").X for i in range(len(columns))]
        objective = float(model.ObjVal)
    description = _netlib.render_description(source, epsilon, delta)
    return objective, solution, description


def build_records(source_dir: Path = SOURCE_DIR) -> list[dict]:
    """Build all frozen recipes and recompute their exact robust counterparts."""
    from . import _external_models as models

    specs = json.loads((source_dir / "external_specs.json").read_text(encoding="utf-8"))
    records = []
    for spec in specs:
        record = copy.deepcopy(spec)
        recipe = record.pop("recipe")
        kind = recipe["kind"]
        if kind == "drug":
            r1 = 100000.0 / 113.93
            d1 = 0.0199 * r1
            objective, solution = 100.0 * r1 - 5500.0 * d1, [r1, 0.0, d1, 0.0]
            description = models.DRUG_DESCRIPTION
        elif kind == "portfolio":
            uncertainty, parameter = recipe["uncertainty"], recipe["parameter"]
            objective, solution = models.solve_portfolio(uncertainty, parameter)
            description = models.portfolio_latex(uncertainty, parameter)
        elif kind == "ldf":
            case, uncertainty, radius = recipe["case"], recipe["uncertainty"], recipe["radius"]
            if recipe["layout"] == "rows":
                if recipe["model"] == "lp":
                    objective, solution = models.solve_ldf_rows_lp(case, uncertainty, radius)
                    description = models.serialize_ldf_lp("T100", case, uncertainty, radius)
                else:
                    objective, solution = models.solve_ldf_rows_milp(case, uncertainty, radius)
                    description = models.serialize_ldf_milp(case, uncertainty, radius)
            elif recipe["model"] == "lp":
                objective, solution = models.solve_ldf_compact_lp(case, uncertainty, radius)
                description = models.ldf_lp_latex(case, uncertainty, radius)
            else:
                objective, solution = models.solve_ldf_compact_milp(case, uncertainty)
                description = models.ldf_milp_latex(case, uncertainty)
        elif kind == "mpp":
            uncertainty, parameter = recipe["uncertainty"], recipe["parameter"]
            objective, solution = models.solve_lf2012(uncertainty, parameter)
            description = models.lf2012_latex(uncertainty, parameter)
        elif kind == "selection":
            n, p, gamma, nominal, deviations = models.parse_dat(source_dir / recipe["file"])
            objective, solution, _ = models.solve_selection(nominal, deviations, p, gamma)
            description = models.selection_latex(nominal, deviations, n, p, gamma)
        elif kind == "netlib":
            objective, solution, description = _solve_netlib(
                source_dir / recipe["file"], Decimal(recipe["epsilon"]), Decimal(recipe["delta"])
            )
        else:
            raise ValueError(f"Unknown recipe kind: {kind}")
        record.update(description=description, solution=solution, optimal_value=float(objective))
        if len(solution) != record["n_vars"]:
            raise ValueError(f"Solution dimension mismatch for {record['id']}")
        records.append(record)
    if len(records) != 32 or len({r["id"] for r in records}) != 32:
        raise ValueError("The external recipes must contain exactly 32 distinct instances.")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output JSON dataset.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    args = parser.parse_args()
    records = build_records(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
