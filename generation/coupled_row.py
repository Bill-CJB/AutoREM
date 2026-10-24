
"""
Generate Random-style robust LP instances with one coupled pair of rows.

Each instance starts from ``synthetic.generate_random_params`` and keeps
the Random split's exact row structure: among the objective plus ``n``
constraint rows, exactly one row is deterministic and the other ``n`` rows
are uncertain.  Two of those existing uncertain rows are sampled uniformly
without replacement; row 0 is the objective and rows 1..n are constraints.
The selected pair then shares one primitive uncertainty vector and one set:

    r_i(xi) = r_i_nom + I xi
    r_j(xi) = r_j_nom + I xi
    xi in U_shared

The shared family is sampled from Box, Budgeted, and Polyhedral uncertainty.
All remaining uncertain rows retain independent Random-style uncertainty.
The original deterministic row (including a possible equality) is preserved,
RHS uncertainty is disabled, and every record uses the canonical T111
description template.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .synthetic import (
        _x0_poly_sample,
        generate_feasible_lp,
        generate_random_params,
    )
except ImportError:
    from generation.synthetic import (
        _x0_poly_sample,
        generate_feasible_lp,
        generate_random_params,
    )


FAMILIES = ("box", "budget", "poly")
FAMILY_CODE = {"box": 1, "budget": 2, "poly": 3}
FAMILY_LABEL = {1: "Box", 2: "Budgeted", 3: "Polyhedral"}
CANONICAL_TEMPLATE_PARAM = [1, 1, 1]


@dataclass
class SharedUncertaintySpec:
    family: str
    gamma: float | None = None
    poly_constraints: list[tuple[np.ndarray, float, int]] | None = None
    poly_lower: float | None = None
    poly_upper: float | None = None
    poly_witness: np.ndarray | None = None


@dataclass
class CoupledRandomSpec:
    sample_id: str
    params: dict[str, Any]
    coupled_units: tuple[int, int]  
    shared: SharedUncertaintySpec
    original_rhs: np.ndarray
    backbone_uncertainty_types: tuple[int, ...]

    @property
    def n_vars(self) -> int:
        return int(self.params["n_vars"])


def _fmt_number(value: float) -> str:
    value = float(value)
    if abs(value) < 5e-10:
        value = 0.0
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _fmt_vector(values: Any) -> str:
    flat = np.asarray(values, dtype=float).reshape(-1)
    return (
        "\\begin{bmatrix}"
        + " \\\\ ".join(_fmt_number(value) for value in flat)
        + "\\end{bmatrix}"
    )


def _fmt_matrix(values: Any) -> str:
    matrix = np.asarray(values, dtype=float)
    rows = [
        " & ".join(_fmt_number(value) for value in row)
        for row in matrix
    ]
    return "\\begin{bmatrix}" + " \\\\ ".join(rows) + "\\end{bmatrix}"


def _ineq_symbol(direction: int) -> str:
    if int(direction) == 1:
        return "\\le"
    if int(direction) == -1:
        return "\\ge"
    return "="


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _poly_constraint_matrices(
    constraints: list[tuple[Any, float, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inequalities: list[np.ndarray] = []
    inequality_rhs: list[float] = []
    equalities: list[np.ndarray] = []
    equality_rhs: list[float] = []

    for coeffs, rhs, direction in constraints:
        row = np.asarray(coeffs, dtype=float)
        if int(direction) == 1:
            inequalities.append(row)
            inequality_rhs.append(float(rhs))
        elif int(direction) == -1:
            inequalities.append(-row)
            inequality_rhs.append(-float(rhs))
        else:
            equalities.append(row)
            equality_rhs.append(float(rhs))

    dim = len(np.asarray(constraints[0][0]).reshape(-1)) if constraints else 0
    d_mat = (
        np.vstack(inequalities)
        if inequalities
        else np.empty((0, dim), dtype=float)
    )
    d_vec = np.asarray(inequality_rhs, dtype=float)
    e_mat = (
        np.vstack(equalities)
        if equalities
        else np.empty((0, dim), dtype=float)
    )
    e_vec = np.asarray(equality_rhs, dtype=float)
    return d_mat, d_vec, e_mat, e_vec


def _sample_shared_uncertainty(
    params: dict[str, Any],
    family: str,
) -> SharedUncertaintySpec:
    n_vars = int(params["n_vars"])
    if family == "box":
        return SharedUncertaintySpec(family=family)
    if family == "budget":
        return SharedUncertaintySpec(
            family=family,
            gamma=round(random.uniform(0, n_vars), 1),
        )
    if family != "poly":
        raise ValueError(f"Unsupported shared family: {family}")

    poly_range = float(params["poly_range"])
    poly_lower = float(params["poly_lower"])
    poly_upper = float(params["poly_upper"])
    witness = np.asarray(
        _x0_poly_sample(
            poly_lower,
            poly_upper,
            poly_range,
            n_vars,
        ),
        dtype=float,
    )
    a_poly, b_poly, directions, _ = generate_feasible_lp(
        n=n_vars,
        x0=witness,
        base_range=(-poly_range, poly_range),
    )
    constraints = [
        (
            np.asarray(a_poly[index], dtype=float),
            float(b_poly[index]),
            int(directions[index]),
        )
        for index in range(n_vars - 1)
    ]
    return SharedUncertaintySpec(
        family=family,
        poly_constraints=constraints,
        poly_lower=poly_lower,
        poly_upper=poly_upper,
        poly_witness=witness,
    )


def _make_spec(
    n_vars: int,
    shared_family: str,
    sample_id: str,
) -> CoupledRandomSpec:
    params = generate_random_params(n_vars)
    backbone_types = (
        int(params["objective_uncertainty_type"]),
        *[int(value) for value in params["constraint_types"]],
    )
    eligible_units = [
        unit_id
        for unit_id, type_code in enumerate(backbone_types)
        if type_code != 0
    ]
    if len(eligible_units) != n_vars:
        raise RuntimeError(
            "The Random backbone must provide exactly n uncertain rows."
        )
    coupled_units = tuple(sorted(random.sample(eligible_units, 2)))

    original_rhs = np.asarray(
        [constraint["b"] for constraint in params["constraint_params"]],
        dtype=float,
    )
    shared = _sample_shared_uncertainty(params, shared_family)
    shared_code = FAMILY_CODE[shared_family]

    for unit_id in coupled_units:
        if unit_id == 0:
            params["objective_uncertainty_type"] = shared_code
            params["uncertainty_index"][0] = list(range(n_vars))
            unit_params = params["objective_uncertainty_params"]
        else:
            row = unit_id - 1
            params["constraint_types"][row] = shared_code
            params["uncertainty_index"][unit_id] = list(range(n_vars))
            unit_params = params["constraint_params"][row]
        for key in (
            "deviation",
            "gamma",
            "poly_constraints",
            "b_uncertain",
            "b_deviation",
        ):
            unit_params.pop(key, None)

    return CoupledRandomSpec(
        sample_id=sample_id,
        params=params,
        coupled_units=(
            int(coupled_units[0]),
            int(coupled_units[1]),
        ),
        shared=shared,
        original_rhs=original_rhs,
        backbone_uncertainty_types=backbone_types,
    )


def _build_poly_set(
    variable: Any,
    constraints: list[tuple[Any, float, int]],
    lower: float,
    upper: float,
) -> tuple[Any, ...]:
    conditions: list[Any] = []
    for coeffs, rhs, direction in constraints:
        expression = np.asarray(coeffs, dtype=float) @ variable
        if int(direction) == 1:
            conditions.append(expression <= float(rhs))
        elif int(direction) == -1:
            conditions.append(expression >= float(rhs))
        else:
            conditions.append(expression == float(rhs))
    conditions.extend(
        [
            variable >= float(lower),
            variable <= float(upper),
        ]
    )
    return tuple(conditions)


def _build_shared_set(
    model: ro.Model,
    spec: CoupledRandomSpec,
) -> tuple[Any, tuple[Any, ...]]:
    xi = model.rvar(spec.n_vars)
    if spec.shared.family == "box":
        uncertainty_set = (rso.norm(xi, np.inf) <= 1,)
    elif spec.shared.family == "budget":
        uncertainty_set = (
            rso.norm(xi, np.inf) <= 1,
            rso.norm(xi, 1) <= float(spec.shared.gamma),
        )
    else:
        uncertainty_set = _build_poly_set(
            xi,
            spec.shared.poly_constraints or [],
            float(spec.shared.poly_lower),
            float(spec.shared.poly_upper),
        )
    return xi, uncertainty_set


def _build_independent_set(
    model: ro.Model,
    type_code: int,
    unit_params: dict[str, Any],
    dimension: int,
    params: dict[str, Any],
) -> tuple[Any, tuple[Any, ...]]:
    perturbation = model.rvar(dimension)
    if type_code == 1:
        uncertainty_set = (rso.norm(perturbation, np.inf) <= 1,)
    elif type_code == 2:
        uncertainty_set = (
            rso.norm(perturbation, np.inf) <= 1,
            rso.norm(perturbation, 1) <= float(unit_params["gamma"]),
        )
    elif type_code == 3:
        uncertainty_set = _build_poly_set(
            perturbation,
            unit_params["poly_constraints"],
            float(params["poly_lower"]),
            float(params["poly_upper"]),
        )
    else:
        raise ValueError(f"Independent uncertainty type must be 1--3, got {type_code}.")
    return perturbation, uncertainty_set


def _independent_uncertain_expression(
    x: Any,
    nominal_coeffs: Any,
    indices: list[int],
    perturbation: Any,
    type_code: int,
    unit_params: dict[str, Any],
) -> Any:
    nominal = np.asarray(nominal_coeffs, dtype=float)
    position = {int(index): offset for offset, index in enumerate(indices)}
    deviations = unit_params.get("deviation", [])
    terms: list[Any] = []

    for index, nominal_value in enumerate(nominal):
        if index not in position:
            coefficient = float(nominal_value)
        else:
            offset = position[index]
            if type_code in (1, 2):
                coefficient = (
                    float(nominal_value)
                    + float(deviations[offset]) * perturbation[offset]
                )
            else:
                coefficient = float(nominal_value) + perturbation[offset]
        terms.append(coefficient * x[index])
    return sum(terms)


def solve_spec(spec: CoupledRandomSpec) -> dict[str, Any]:
    global ro, rso
    from rsome import ro
    import rsome as rso

    """Solve the exact shared-factor model with all other Random units intact."""
    params = spec.params
    n_vars = spec.n_vars
    model = ro.Model()
    x = model.dvar(n_vars)

    
    
    xi_shared, shared_set = _build_shared_set(model, spec)

    coupled = set(spec.coupled_units)
    objective_type = int(params["objective_uncertainty_type"])
    objective_indices = [int(index) for index in params["uncertainty_index"][0]]
    objective_params = params["objective_uncertainty_params"]
    if 0 in coupled:
        objective = sum(
            (
                float(params["objective_coeffs"][index])
                + xi_shared[index]
            )
            * x[index]
            for index in range(n_vars)
        )
        if bool(params["objective_direction"]):
            model.maxmin(objective, shared_set)
        else:
            model.minmax(objective, shared_set)
    elif objective_type == 0:
        objective = np.asarray(params["objective_coeffs"], dtype=float) @ x
        if bool(params["objective_direction"]):
            model.max(objective)
        else:
            model.min(objective)
    else:
        objective_z, objective_set = _build_independent_set(
            model,
            objective_type,
            objective_params,
            len(objective_indices),
            params,
        )
        objective = _independent_uncertain_expression(
            x,
            params["objective_coeffs"],
            objective_indices,
            objective_z,
            objective_type,
            objective_params,
        )
        if bool(params["objective_direction"]):
            model.maxmin(objective, objective_set)
        else:
            model.minmax(objective, objective_set)

    for row, constraint in enumerate(params["constraint_params"]):
        unit_id = row + 1
        nominal = np.asarray(constraint["nominal_coeffs"], dtype=float)
        rhs = float(constraint["b"])
        direction = int(constraint["direction"])

        if unit_id in coupled:
            
            expression = sum(
                (float(nominal[index]) + xi_shared[index]) * x[index]
                for index in range(n_vars)
            )
            robust_constraint = (
                expression <= rhs if direction == 1 else expression >= rhs
            )
            model.st(robust_constraint.forall(shared_set))
            continue

        type_code = int(params["constraint_types"][row])
        if type_code == 0:
            if direction == 1:
                model.st(nominal @ x <= rhs)
            elif direction == -1:
                model.st(nominal @ x >= rhs)
            else:
                model.st(nominal @ x == rhs)
            continue

        indices = [
            int(index) for index in params["uncertainty_index"][row + 1]
        ]
        row_z, row_set = _build_independent_set(
            model,
            type_code,
            constraint,
            len(indices),
            params,
        )
        expression = _independent_uncertain_expression(
            x,
            nominal,
            indices,
            row_z,
            type_code,
            constraint,
        )
        robust_constraint = (
            expression <= rhs if direction == 1 else expression >= rhs
        )
        model.st(robust_constraint.forall(row_set))

    model.st(
        x >= float(params["x_lower"]),
        x <= float(params["x_upper"]),
    )

    try:
        model.solve()
        solution = np.asarray(x.get(), dtype=float)
        optimal_value = float(model.get())
        return {
            "status": "optimal",
            "solution": solution.tolist(),
            "optimal_value": optimal_value,
        }
    except Exception as exc:  
        message = str(exc)
        lowered = message.lower()
        if "unbounded" in lowered or "status: 3" in lowered:
            status = "unbounded"
        elif "infeasible" in lowered or "status: 2" in lowered:
            status = "infeasible"
        else:
            status = "error"
        return {"status": status, "error": message}


def _full_deviation_vector(
    n_vars: int,
    indices: list[int],
    unit_params: dict[str, Any],
) -> np.ndarray:
    vector = np.zeros(n_vars, dtype=float)
    deviations = unit_params["deviation"]
    for offset, index in enumerate(indices):
        vector[int(index)] = float(deviations[offset])
    return vector


def _unit_symbols(prefix: str, type_code: int) -> tuple[str, str]:
    if type_code == 3:
        variable = f"\\boldsymbol{{\\delta}}_{{{prefix}}}"
    else:
        variable = f"\\boldsymbol{{\\xi}}_{{{prefix}}}"
    suffix = {1: "Box", 2: "Bud", 3: "P"}[type_code]
    set_name = f"\\mathcal{{U}}_{{{prefix},{suffix}}}"
    return variable, set_name


def _full_poly_matrices(
    n_vars: int,
    indices: list[int],
    constraints: list[tuple[Any, float, int]],
    poly_lower: float,
    poly_upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Embed the Random generator's reduced polyhedron in R^n.

    This mirrors the original T111 renderer: selected perturbation coordinates
    receive the generated polyhedral rows and component bounds, while every
    unselected coordinate is fixed to zero through an equality.
    """
    inequalities: list[np.ndarray] = []
    inequality_rhs: list[float] = []
    equalities: list[np.ndarray] = []
    equality_rhs: list[float] = []

    for reduced_coeffs, rhs, direction in constraints:
        row = np.zeros(n_vars, dtype=float)
        for offset, index in enumerate(indices):
            row[int(index)] = float(reduced_coeffs[offset])
        if int(direction) == 1:
            inequalities.append(row)
            inequality_rhs.append(float(rhs))
        elif int(direction) == -1:
            inequalities.append(-row)
            inequality_rhs.append(-float(rhs))
        else:
            equalities.append(row)
            equality_rhs.append(float(rhs))

    for index in indices:
        upper_row = np.zeros(n_vars, dtype=float)
        upper_row[int(index)] = 1.0
        inequalities.append(upper_row)
        inequality_rhs.append(float(poly_upper))

        lower_row = np.zeros(n_vars, dtype=float)
        lower_row[int(index)] = -1.0
        inequalities.append(lower_row)
        inequality_rhs.append(-float(poly_lower))

    selected = set(int(index) for index in indices)
    for index in range(n_vars):
        if index in selected:
            continue
        equality_row = np.zeros(n_vars, dtype=float)
        equality_row[index] = 1.0
        equalities.append(equality_row)
        equality_rhs.append(0.0)

    return (
        np.vstack(inequalities),
        np.asarray(inequality_rhs, dtype=float),
        np.vstack(equalities)
        if equalities
        else np.empty((0, n_vars), dtype=float),
        np.asarray(equality_rhs, dtype=float),
    )


def _render_set_definition(
    prefix: str,
    type_code: int,
    unit_params: dict[str, Any],
    indices: list[int],
    n_vars: int,
    poly_lower: float,
    poly_upper: float,
) -> list[str]:
    variable, set_name = _unit_symbols(prefix, type_code)
    if type_code == 1:
        return [
            f"& {set_name}=\\{{{variable}\\in\\mathbb{{R}}^{{{n_vars}}}:"
            f"\\|{variable}\\|_\\infty\\le1\\}}. \\\\"
        ]
    if type_code == 2:
        return [
            f"& {set_name}=\\{{{variable}\\in\\mathbb{{R}}^{{{n_vars}}}:"
            f"\\|{variable}\\|_\\infty\\le1,"
            f"\\|{variable}\\|_1\\le{_fmt_number(unit_params['gamma'])}\\}}. \\\\"
        ]

    d_mat, d_vec, e_mat, e_vec = _full_poly_matrices(
        n_vars,
        indices,
        unit_params["poly_constraints"],
        poly_lower,
        poly_upper,
    )
    clauses = [
        f"\\mathbf{{D}}_{{{prefix}}}{variable}\\le\\mathbf{{d}}_{{{prefix}}}"
    ]
    if e_mat.shape[0]:
        clauses.append(f"\\mathbf{{E}}_{{{prefix}}}{variable}=\\mathbf{{e}}_{{{prefix}}}")
    lines = [
        f"& {set_name}=\\{{{variable}\\in\\mathbb{{R}}^{{{n_vars}}}:"
        + ",\\ ".join(clauses)
        + "\\}. \\\\"
    ]
    lines.append(
        f"& \\mathbf{{D}}_{{{prefix}}}={_fmt_matrix(d_mat)},"
        f"\\quad\\mathbf{{d}}_{{{prefix}}}={_fmt_vector(d_vec)}. \\\\"
    )
    if e_mat.shape[0]:
        lines.append(
            f"& \\mathbf{{E}}_{{{prefix}}}={_fmt_matrix(e_mat)},"
            f"\\quad\\mathbf{{e}}_{{{prefix}}}={_fmt_vector(e_vec)}. \\\\"
        )
    return lines


def _render_shared_set(spec: CoupledRandomSpec) -> list[str]:
    variable = "\\boldsymbol{\\xi}_{\\mathrm{shared}}"
    set_name = "\\mathcal{U}_{\\mathrm{shared}}"
    n_vars = spec.n_vars
    if spec.shared.family == "box":
        return [
            f"& {set_name}=\\{{{variable}\\in\\mathbb{{R}}^{{{n_vars}}}:"
            f"\\|{variable}\\|_\\infty\\le1\\}}. \\\\"
        ]
    if spec.shared.family == "budget":
        return [
            f"& {set_name}=\\{{{variable}\\in\\mathbb{{R}}^{{{n_vars}}}:"
            f"\\|{variable}\\|_\\infty\\le1,"
            f"\\|{variable}\\|_1\\le{_fmt_number(spec.shared.gamma)}\\}}. \\\\"
        ]

    constraints = spec.shared.poly_constraints or []
    d_mat, d_vec, e_mat, e_vec = _poly_constraint_matrices(constraints)
    clauses = []
    if d_mat.shape[0]:
        clauses.append("\\mathbf{D}_{s}\\boldsymbol{\\xi}_{\\mathrm{shared}}\\le\\mathbf{d}_{s}")
    if e_mat.shape[0]:
        clauses.append("\\mathbf{E}_{s}\\boldsymbol{\\xi}_{\\mathrm{shared}}=\\mathbf{e}_{s}")
    clauses.append(
        f"{_fmt_number(spec.shared.poly_lower)}\\mathbf{{1}}"
        "\\le\\boldsymbol{\\xi}_{\\mathrm{shared}}"
        f"\\le{_fmt_number(spec.shared.poly_upper)}\\mathbf{{1}}"
    )
    lines = [
        f"& {set_name}=\\{{{variable}\\in\\mathbb{{R}}^{{{n_vars}}}:"
        + ",\\ ".join(clauses)
        + "\\}. \\\\"
    ]
    if d_mat.shape[0]:
        lines.append(
            f"& \\mathbf{{D}}_s={_fmt_matrix(d_mat)},"
            f"\\quad\\mathbf{{d}}_s={_fmt_vector(d_vec)}. \\\\"
        )
    if e_mat.shape[0]:
        lines.append(
            f"& \\mathbf{{E}}_s={_fmt_matrix(e_mat)},"
            f"\\quad\\mathbf{{e}}_s={_fmt_vector(e_vec)}. \\\\"
        )
    return lines


def render_description(spec: CoupledRandomSpec) -> str:
    """Render a dedicated T111 problem with exactly one shared-row pair."""
    params = spec.params
    n_vars = spec.n_vars
    coupled = set(spec.coupled_units)
    shared_code = FAMILY_CODE[spec.shared.family]
    objective_type = int(params["objective_uncertainty_type"])
    outer = "\\text{Maximize}" if bool(params["objective_direction"]) else "\\text{Minimize}"
    inner = "\\min" if bool(params["objective_direction"]) else "\\max"

    lines = ["\\begin{align*}"]
    if 0 in coupled:
        lines.append(
            f"& {outer}\\quad {inner}_{{\\boldsymbol{{\\xi}}_{{\\mathrm{{shared}}}}"
            "\\in\\mathcal{U}_{\\mathrm{shared}}}"
            f"(\\bar{{\\mathbf{{c}}}}+\\mathbf{{I}}_{{{n_vars}}}"
            "\\boldsymbol{\\xi}_{\\mathrm{shared}})^\\top\\mathbf{x}"
            f" && \\text{{(Shared-factor {FAMILY_LABEL[shared_code]} "
            "Uncertainty Objective)} \\\\"
        )
    elif objective_type == 0:
        lines.append(
            f"& {outer}\\quad \\bar{{\\mathbf{{c}}}}^\\top\\mathbf{{x}}"
            " && \\text{(Deterministic Objective)} \\\\"
        )
    else:
        objective_variable, objective_set = _unit_symbols(
            "c",
            objective_type,
        )
        if objective_type in (1, 2):
            objective_expression = (
                "(\\bar{\\mathbf{c}}+\\operatorname{diag}(\\hat{\\mathbf{c}})"
                f"{objective_variable})^\\top\\mathbf{{x}}"
            )
        else:
            objective_expression = (
                f"(\\bar{{\\mathbf{{c}}}}+{objective_variable})^\\top"
                "\\mathbf{x}"
            )
        lines.append(
            f"& {outer}\\quad {inner}_{{{objective_variable}\\in{objective_set}}}"
            f"{objective_expression}"
            f" && \\text{{({FAMILY_LABEL[objective_type]} "
            "Uncertainty Objective)} \\\\"
        )
    lines.append("& \\text{subject to} \\\\")

    for row, constraint in enumerate(params["constraint_params"]):
        unit_id = row + 1
        direction = int(constraint["direction"])
        type_code = int(params["constraint_types"][row])
        symbol = _ineq_symbol(direction)
        rhs = _fmt_number(constraint["b"])
        if unit_id in coupled:
            worst_case = "\\max" if direction == 1 else "\\min"
            lines.append(
                f"& {worst_case}_{{\\boldsymbol{{\\xi}}_{{\\mathrm{{shared}}}}"
                "\\in\\mathcal{U}_{\\mathrm{shared}}}"
                f"(\\bar{{\\mathbf{{a}}}}_{{{unit_id}}}"
                f"+\\mathbf{{I}}_{{{n_vars}}}\\boldsymbol{{\\xi}}_{{\\mathrm{{shared}}}})^\\top"
                f"\\mathbf{{x}} {symbol} {rhs}"
                f" && \\text{{(Shared-factor {FAMILY_LABEL[shared_code]}"
                " Uncertainty Constraint)} \\\\"
            )
        elif type_code == 0:
            nominal_symbol = (
                f"\\mathbf{{a}}_{{{unit_id}}}"
                if direction == 0
                else f"\\bar{{\\mathbf{{a}}}}_{{{unit_id}}}"
            )
            label = (
                "Deterministic Equality Constraint"
                if direction == 0
                else "Deterministic Constraint"
            )
            lines.append(
                f"& {nominal_symbol}^\\top\\mathbf{{x}} {symbol} {rhs}"
                f" && \\text{{({label})}} \\\\"
            )
        else:
            variable, set_name = _unit_symbols(f"a{unit_id}", type_code)
            worst_case = "\\max" if direction == 1 else "\\min"
            if type_code in (1, 2):
                uncertain_expression = (
                    f"(\\bar{{\\mathbf{{a}}}}_{{{unit_id}}}"
                    f"+\\operatorname{{diag}}(\\hat{{\\mathbf{{a}}}}_{{{unit_id}}})"
                    f"{variable})^\\top\\mathbf{{x}}"
                )
            else:
                uncertain_expression = (
                    f"(\\bar{{\\mathbf{{a}}}}_{{{unit_id}}}"
                    f"+{variable})^\\top\\mathbf{{x}}"
                )
            lines.append(
                f"& {worst_case}_{{{variable}\\in{set_name}}}"
                f"{uncertain_expression} {symbol} {rhs}"
                f" && \\text{{({FAMILY_LABEL[type_code]} Uncertainty Constraint)}} \\\\"
            )

    pair_labels = [
        "the objective row" if unit_id == 0 else f"constraint row {unit_id}"
        for unit_id in spec.coupled_units
    ]
    lines.extend(
        [
            "& \\text{where the same primitive vector }"
            "\\boldsymbol{\\xi}_{\\mathrm{shared}}"
            f"\\text{{ drives {pair_labels[0]} and {pair_labels[1]}}}. \\\\",
            f"& \\mathbf{{x}}\\in\\mathbb{{R}}^{{{n_vars}}},\\quad"
            f"{_fmt_number(params['x_lower'])}\\le x_j"
            f"\\le{_fmt_number(params['x_upper'])},"
            f"\\quad j=1,\\ldots,{n_vars}. \\\\",
            f"& \\bar{{\\mathbf{{c}}}}={_fmt_vector(params['objective_coeffs'])}. \\\\",
        ]
    )

    objective_indices = [
        int(index) for index in params["uncertainty_index"][0]
    ]
    if 0 in coupled:
        lines.append(
            f"& \\mathbf{{P}}_c=\\mathbf{{I}}_{{{n_vars}}}. \\\\"
        )
    elif objective_type in (1, 2):
        objective_deviation = _full_deviation_vector(
            n_vars,
            objective_indices,
            params["objective_uncertainty_params"],
        )
        lines.append(
            f"& \\hat{{\\mathbf{{c}}}}={_fmt_vector(objective_deviation)}. \\\\"
        )

    for row, constraint in enumerate(params["constraint_params"]):
        unit_id = row + 1
        type_code = int(params["constraint_types"][row])
        nominal_symbol = (
            f"\\bar{{\\mathbf{{a}}}}_{{{unit_id}}}"
            if int(constraint["direction"]) != 0
            else f"\\mathbf{{a}}_{{{unit_id}}}"
        )
        lines.append(
            f"& {nominal_symbol}={_fmt_vector(constraint['nominal_coeffs'])},"
            f"\\quad b_{{{unit_id}}}={_fmt_number(constraint['b'])}. \\\\"
        )
        if unit_id in coupled:
            lines.append(
                f"& \\mathbf{{P}}_{{a{unit_id}}}=\\mathbf{{I}}_{{{n_vars}}}. \\\\"
            )
        elif type_code in (1, 2):
            indices = [
                int(index) for index in params["uncertainty_index"][unit_id]
            ]
            deviation = _full_deviation_vector(
                n_vars,
                indices,
                constraint,
            )
            lines.append(
                f"& \\hat{{\\mathbf{{a}}}}_{{{unit_id}}}"
                f"={_fmt_vector(deviation)}. \\\\"
            )

    lines.extend(_render_shared_set(spec))
    if 0 not in coupled and objective_type != 0:
        lines.extend(
            _render_set_definition(
                "c",
                objective_type,
                params["objective_uncertainty_params"],
                objective_indices,
                n_vars,
                float(params["poly_lower"]),
                float(params["poly_upper"]),
            )
        )
    for row, constraint in enumerate(params["constraint_params"]):
        unit_id = row + 1
        type_code = int(params["constraint_types"][row])
        if unit_id in coupled or type_code == 0:
            continue
        indices = [
            int(index) for index in params["uncertainty_index"][unit_id]
        ]
        lines.extend(
            _render_set_definition(
                f"a{unit_id}",
                type_code,
                constraint,
                indices,
                n_vars,
                float(params["poly_lower"]),
                float(params["poly_upper"]),
            )
        )

    lines.append("\\end{align*}")
    return "\n".join(lines)


def _shared_metadata(shared: SharedUncertaintySpec) -> dict[str, Any]:
    metadata: dict[str, Any] = {"family": shared.family}
    if shared.family == "budget":
        metadata["gamma"] = shared.gamma
    elif shared.family == "poly":
        metadata.update(
            {
                "poly_constraints": shared.poly_constraints,
                "poly_lower": shared.poly_lower,
                "poly_upper": shared.poly_upper,
                "poly_witness": shared.poly_witness,
            }
        )
    return _json_safe(metadata)


def _record_from_spec(
    spec: CoupledRandomSpec,
    solve_result: dict[str, Any],
) -> dict[str, Any]:
    params = spec.params
    type_list = [
        int(params["objective_uncertainty_type"]),
        *[int(value) for value in params["constraint_types"]],
    ]
    coupled_units = [int(unit_id) for unit_id in spec.coupled_units]
    independent_units = [
        unit_id
        for unit_id, type_code in enumerate(type_list)
        if type_code != 0 and unit_id not in spec.coupled_units
    ]
    independent_constraint_rows = [
        unit_id for unit_id in independent_units if unit_id != 0
    ]
    deterministic_units = [
        unit_id
        for unit_id, type_code in enumerate(type_list)
        if type_code == 0
    ]
    equality_rows = [
        row + 1
        for row, constraint in enumerate(params["constraint_params"])
        if int(constraint["direction"]) == 0
    ]
    coupled_labels = [
        "objective" if unit_id == 0 else f"constraint_{unit_id}"
        for unit_id in coupled_units
    ]
    pair_kind = (
        "objective_constraint"
        if 0 in spec.coupled_units
        else "constraint_constraint"
    )
    return {
        "id": spec.sample_id,
        "n_vars": spec.n_vars,
        "n_constraints": int(params["n_constraints"]),
        "template_param": CANONICAL_TEMPLATE_PARAM.copy(),
        "setting": "random_backbone_coupled_row_shared_primitive",
        "description": render_description(spec),
        "uncertainty_type": type_list,
        "solution": solve_result["solution"],
        "optimal_value": solve_result["optimal_value"],
        "coupled_rows": coupled_units,
        "coupled_unit_ids": coupled_units,
        "coupled_unit_labels": coupled_labels,
        "row_indexing": "0=objective, 1..n=constraint rows",
        "shared_uncertainty": True,
        "uncertainty_family": spec.shared.family,
        "generation_metadata": {
            "base_generator": "synthetic.generate_random_params",
            "coupling_mode": "shared_primitive",
            "row_indexing": "0=objective, 1..n=constraint rows",
            "coupled_rows": coupled_units,
            "coupled_row_labels": coupled_labels,
            "pair_kind": pair_kind,
            "backbone_uncertainty_types": list(
                spec.backbone_uncertainty_types
            ),
            "independent_uncertain_units": independent_units,
            "independent_uncertain_constraint_rows": (
                independent_constraint_rows
            ),
            "deterministic_units": deterministic_units,
            "deterministic_equality_rows": equality_rows,
            "objective_uncertainty_type": int(
                params["objective_uncertainty_type"]
            ),
            "objective_is_coupled": 0 in spec.coupled_units,
            "num_uncertain_units": sum(
                type_code != 0 for type_code in type_list
            ),
            "num_deterministic_units": sum(
                type_code == 0 for type_code in type_list
            ),
            "shared_factor_dim": spec.n_vars,
            "shared_factor_symbol": "xi_shared",
            "shared_set": _shared_metadata(spec.shared),
            "loading_structure": "identity",
            "loading_value": 1.0,
            "rhs_uncertainty": False,
            "template": "T111",
        },
    }


def _validate_independent_unit(
    type_code: int,
    unit_params: dict[str, Any],
    indices: list[int],
) -> None:
    if type_code not in (1, 2, 3):
        raise AssertionError(f"Expected uncertainty type 1--3, got {type_code}.")
    if len(indices) < 2:
        raise AssertionError("Every Random-style uncertain unit needs >=2 indices.")
    if type_code in (1, 2):
        if len(unit_params.get("deviation", [])) != len(indices):
            raise AssertionError("Deviation length does not match uncertainty indices.")
        if type_code == 2 and "gamma" not in unit_params:
            raise AssertionError("Budgeted unit is missing gamma.")
    elif len(unit_params.get("poly_constraints", [])) != len(indices) - 1:
        raise AssertionError("Polyhedral unit has the wrong number of constraints.")


def _validate_spec(spec: CoupledRandomSpec) -> None:
    params = spec.params
    n_vars = spec.n_vars
    if int(params["n_constraints"]) != n_vars:
        raise AssertionError("Random backbone must have n constraints.")
    if len(spec.coupled_units) != 2 or len(set(spec.coupled_units)) != 2:
        raise AssertionError("Exactly two distinct coupled rows are required.")

    type_list = [
        int(params["objective_uncertainty_type"]),
        *[int(value) for value in params["constraint_types"]],
    ]
    backbone_types = list(spec.backbone_uncertainty_types)
    if len(type_list) != n_vars + 1 or len(backbone_types) != n_vars + 1:
        raise AssertionError("Row-type lists must contain objective plus n constraints.")
    if sum(type_code == 0 for type_code in type_list) != 1:
        raise AssertionError("The final instance must keep one deterministic row.")
    if sum(type_code != 0 for type_code in type_list) != n_vars:
        raise AssertionError("The final instance must keep n uncertain rows.")
    if sum(type_code == 0 for type_code in backbone_types) != 1:
        raise AssertionError("The Random backbone must have one deterministic row.")

    deterministic_unit = type_list.index(0)
    if deterministic_unit != backbone_types.index(0):
        raise AssertionError("Coupling changed the Random deterministic row.")

    shared_code = FAMILY_CODE[spec.shared.family]
    for unit_id in spec.coupled_units:
        if not (0 <= unit_id <= n_vars):
            raise AssertionError("Coupled row index is outside 0..n.")
        if backbone_types[unit_id] == 0:
            raise AssertionError("A deterministic row cannot be coupled.")
        if type_list[unit_id] != shared_code:
            raise AssertionError("Coupled row type does not match shared family.")
    for unit_id in range(n_vars + 1):
        if unit_id not in spec.coupled_units:
            if type_list[unit_id] != backbone_types[unit_id]:
                raise AssertionError("A non-coupled row changed uncertainty type.")

    objective_type = int(params["objective_uncertainty_type"])
    objective_indices = [
        int(index) for index in params["uncertainty_index"][0]
    ]
    if 0 in spec.coupled_units:
        if objective_indices != list(range(n_vars)):
            raise AssertionError("P=I coupled objective must use every coordinate.")
    elif objective_type == 0:
        if objective_indices:
            raise AssertionError("Deterministic objective has uncertainty indices.")
    else:
        _validate_independent_unit(
            objective_type,
            params["objective_uncertainty_params"],
            objective_indices,
        )

    current_rhs = np.asarray(
        [constraint["b"] for constraint in params["constraint_params"]],
        dtype=float,
    )
    if not np.array_equal(current_rhs, spec.original_rhs):
        raise AssertionError("RHS values changed after selecting the coupled pair.")
    for constraint in params["constraint_params"]:
        if bool(constraint.get("b_uncertain", False)):
            raise AssertionError("RHS uncertainty must remain disabled.")
        if "b_deviation" in constraint:
            raise AssertionError("RHS deviation metadata must not be present.")

    for row, constraint in enumerate(params["constraint_params"]):
        unit_id = row + 1
        direction = int(constraint["direction"])
        type_code = int(params["constraint_types"][row])
        indices = [
            int(index) for index in params["uncertainty_index"][unit_id]
        ]
        if direction == 0:
            if type_code != 0 or indices:
                raise AssertionError("Equality rows must remain deterministic.")
            if unit_id in spec.coupled_units:
                raise AssertionError("An equality row cannot be coupled.")
            continue
        if unit_id in spec.coupled_units:
            if type_code != shared_code:
                raise AssertionError("Coupled row type does not match shared family.")
            if indices != list(range(n_vars)):
                raise AssertionError("P=I coupled rows must use every coordinate.")
            continue
        if type_code == 0:
            if indices:
                raise AssertionError("A deterministic constraint has uncertainty indices.")
            continue
        _validate_independent_unit(type_code, constraint, indices)

    if spec.shared.family == "budget":
        if spec.shared.gamma is None or not (0 <= spec.shared.gamma <= n_vars):
            raise AssertionError("Shared budget gamma is outside [0,n].")
    elif spec.shared.family == "poly":
        constraints = spec.shared.poly_constraints or []
        if len(constraints) != n_vars - 1:
            raise AssertionError("Shared polyhedron has the wrong row count.")
        witness = np.asarray(spec.shared.poly_witness, dtype=float)
        if witness.shape != (n_vars,):
            raise AssertionError("Shared polyhedral witness has the wrong shape.")
        lower = float(spec.shared.poly_lower)
        upper = float(spec.shared.poly_upper)
        if np.any(witness < lower - 1e-8) or np.any(witness > upper + 1e-8):
            raise AssertionError("Shared polyhedral witness violates its bounds.")
        for coeffs, rhs, direction in constraints:
            value = float(np.asarray(coeffs, dtype=float) @ witness)
            if int(direction) == 1 and value > float(rhs) + 1e-7:
                raise AssertionError("Shared witness violates a <= poly row.")
            if int(direction) == -1 and value < float(rhs) - 1e-7:
                raise AssertionError("Shared witness violates a >= poly row.")
            if int(direction) == 0 and abs(value - float(rhs)) > 1e-7:
                raise AssertionError("Shared witness violates an equality poly row.")


def generate_dataset(
    num_samples: int = 32,
    var_choices: tuple[int, ...] | list[int] = (2, 3, 4, 5),
    seed: int = 20260726,
    max_attempts_per_sample: int = 200,
) -> list[dict[str, Any]]:
    """
    Generate solver-verified records with balanced dimensions and IID families.

    The requested records are divided as evenly as possible across
    ``var_choices`` (the default 32 records give exactly eight instances for
    each of n=2,3,4,5), then shuffled.  A dedicated selector RNG independently
    draws each shared family before any solver rejection occurs.  Thus failed
    candidate LPs do not bias either target distribution.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    choices = tuple(int(value) for value in var_choices)
    if not choices or any(value < 2 or value > 5 for value in choices):
        raise ValueError("var_choices must contain dimensions from 2 through 5.")

    random.seed(seed)
    np.random.seed(seed)
    selector_rng = random.Random(seed + 104729)
    quotient, remainder = divmod(num_samples, len(choices))
    dimensions = [
        dimension
        for dimension in choices
        for _ in range(quotient)
    ]
    if remainder:
        dimensions.extend(
            selector_rng.sample(list(choices), k=remainder)
        )
    selector_rng.shuffle(dimensions)
    targets = [
        (
            dimension,
            selector_rng.choice(FAMILIES),
        )
        for dimension in dimensions
    ]

    records: list[dict[str, Any]] = []
    for sample_index, (n_vars, shared_family) in enumerate(targets, start=1):
        sample_id = (
            f"coupled_{sample_index:03d}_n{n_vars}_{shared_family}_T111"
        )
        last_error = ""
        for _ in range(max_attempts_per_sample):
            spec = _make_spec(n_vars, shared_family, sample_id)
            _validate_spec(spec)
            solve_result = solve_spec(spec)
            if solve_result["status"] != "optimal":
                last_error = str(
                    solve_result.get("error", solve_result["status"])
                )
                continue

            solution = np.asarray(solve_result["solution"], dtype=float)
            at_bounds = np.isclose(
                solution,
                float(spec.params["x_lower"]),
            ) | np.isclose(
                solution,
                float(spec.params["x_upper"]),
            )
            if np.all(at_bounds):
                last_error = "all decision variables were at their bounds"
                continue
            if not np.isfinite(float(solve_result["optimal_value"])):
                last_error = "non-finite optimal value"
                continue

            records.append(_record_from_spec(spec, solve_result))
            print(
                f"[{len(records)}/{num_samples}] {sample_id}: "
                f"rows={list(spec.coupled_units)}, optimal"
            )
            break
        else:
            raise RuntimeError(
                f"Failed to generate {sample_id} after "
                f"{max_attempts_per_sample} attempts: {last_error}"
            )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Random-style robust LPs with two shared-factor rows "
            "(objective row 0 may be selected)."
        )
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of solver-verified records (default: 32).",
    )
    parser.add_argument(
        "--var-choices",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5],
        help="Dimensions balanced across records, then shuffled (default: 2 3 4 5).",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=200,
        help="Maximum full-instance rejection attempts per requested record.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets")
        / "test_coupled_row_32.json",
    )
    args = parser.parse_args()

    records = generate_dataset(
        num_samples=args.num_samples,
        var_choices=args.var_choices,
        seed=args.seed,
        max_attempts_per_sample=args.max_attempts,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dimension_counts = Counter(record["n_vars"] for record in records)
    shared_family_counts = Counter(
        record["uncertainty_family"] for record in records
    )
    all_unit_type_counts = Counter(
        type_code
        for record in records
        for type_code in record["uncertainty_type"]
    )
    print(f"Saved {len(records)} records to {args.out}")
    print(f"Dimension counts: {dict(sorted(dimension_counts.items()))}")
    print(f"Shared-family counts: {dict(shared_family_counts)}")
    print(f"All-unit type counts: {dict(sorted(all_unit_type_counts.items()))}")


if __name__ == "__main__":
    main()
