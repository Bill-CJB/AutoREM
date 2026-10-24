"""Generate Random-backbone robust LPs with general perturbation matrices.

Each candidate starts from :func:`synthetic.generate_random_params`, so
the objective plus ``n`` constraint rows keep the Random split's exact type
structure: one deterministic unit and ``n`` uncertain units.  Every existing
uncertain unit receives its own independent support-preserving general
loading

    P_u = S_u P_tilde_u,

where ``S_u`` embeds the unit's original ``uncertainty_index`` into the
ambient coefficient vector and ``P_tilde_u`` is dense, non-diagonal, and
full-rank on that active support.  Consequently, coefficients outside the
original Random uncertainty support receive exactly zero perturbation.

The uncertainty family and set parameters of every unit remain unchanged;
only the diagonal/identity coefficient loading on the active support is
replaced.  Different rows continue to use independent factors and sets, RHS
uncertainty is disabled, and the canonical compact T111 representation is
emitted.  The default CLI produces 32 instances: eight each for
``n = 2, 3, 4, 5``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    
    from .synthetic import (
        generate_random_params as generate_random_backbone,
    )
except ImportError:
    
    from generation.synthetic import (  
        generate_random_params as generate_random_backbone,
    )


DETERMINISTIC = 0
BOX = 1
BUDGET = 2
POLYHEDRAL = 3

TYPE_LABELS = {
    DETERMINISTIC: "Deterministic",
    BOX: "Box",
    BUDGET: "Budgeted",
    POLYHEDRAL: "Polyhedral",
}
TYPE_NAMES = {
    BOX: "box",
    BUDGET: "budget",
    POLYHEDRAL: "polyhedral",
}

CANONICAL_TEMPLATE_PARAM = [1, 1, 1]
SETTING_NAME = "general_support_preserving_all_uncertain_P"


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to JSON-native values."""
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _fmt(value: float) -> str:
    """Format numerical data compactly and avoid negative zero."""
    value = float(value)
    if abs(value) < 5e-10:
        value = 0.0
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _vec(values: Iterable[float]) -> str:
    return (
        "\\begin{bmatrix} "
        + " \\\\ ".join(_fmt(value) for value in values)
        + " \\end{bmatrix}"
    )


def _mat(matrix: Iterable[Iterable[float]]) -> str:
    rows = [" & ".join(_fmt(value) for value in row) for row in matrix]
    return "\\begin{bmatrix} " + " \\\\ ".join(rows) + " \\end{bmatrix}"


def _ineq(direction: int) -> str:
    if direction == 1:
        return "\\le"
    if direction == -1:
        return "\\ge"
    return "="


def _generate_dense_active_block(
    row_scales: Sequence[float],
) -> np.ndarray:
    """Sample a dense, non-diagonal, full-rank active loading block.

    The absolute row sum is kept close to the corresponding original Random
    perturbation scale.  For Box/Budget units these scales are the original
    deviation magnitudes; for Polyhedral units they are one because the
    original active loading is the identity.
    """
    scales = np.asarray(row_scales, dtype=float)
    factor_dim = int(scales.size)
    if factor_dim < 2:
        raise ValueError(
            "A non-diagonal General-P block needs at least two active factors."
        )
    if np.any(scales <= 0.0):
        raise ValueError("All active loading row scales must be positive.")

    for _ in range(200):
        
        
        
        rho = float(np.random.uniform(0.25, 0.45))
        random_mixing = np.random.uniform(
            0.2,
            1.0,
            size=(factor_dim, factor_dim),
        )
        random_mixing /= np.sum(random_mixing, axis=1, keepdims=True)
        mixing = (
            (1.0 - rho) * np.eye(factor_dim)
            + rho * random_mixing
        )
        active_block = np.round(scales[:, None] * mixing, 3)

        off_diagonal = active_block.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        if (
            np.count_nonzero(active_block) == active_block.size
            and np.count_nonzero(off_diagonal) > 0
            and np.linalg.matrix_rank(active_block) == factor_dim
            and np.linalg.cond(active_block) <= 50.0
        ):
            return active_block

    raise RuntimeError(
        "Unable to sample a dense full-rank active General-P block."
    )


def _unit_params(params: Dict[str, Any], unit_id: int) -> Dict[str, Any]:
    if unit_id == 0:
        return params["objective_uncertainty_params"]
    return params["constraint_params"][unit_id - 1]


def _unit_nominal(params: Dict[str, Any], unit_id: int) -> Sequence[float]:
    if unit_id == 0:
        return params["objective_coeffs"]
    return params["constraint_params"][unit_id - 1]["nominal_coeffs"]


def _backbone_uncertainty_snapshot(
    params: Dict[str, Any],
    unit_id: int,
    type_code: int,
) -> Dict[str, Any] | None:
    """Copy the Random family/set data that General-P must not modify."""
    if type_code == DETERMINISTIC:
        return None
    unit_params = _unit_params(params, unit_id)
    snapshot: Dict[str, Any] = {
        "uncertainty_family": TYPE_NAMES[type_code],
    }
    if type_code in (BOX, BUDGET):
        snapshot["deviation"] = [
            float(value) for value in unit_params["deviation"]
        ]
        if type_code == BUDGET:
            snapshot["gamma"] = float(unit_params["gamma"])
    else:
        snapshot["poly_constraints"] = unit_params["poly_constraints"]
        snapshot["factor_bounds"] = [
            float(params["poly_lower"]),
            float(params["poly_upper"]),
        ]
    return _json_safe(snapshot)


def generate_random_params(n_vars: int) -> Dict[str, Any]:
    """Generate a Random backbone and generalize every uncertain loading.

    Unit ``0`` is the objective and units ``1..n`` are constraint rows.
    The original family, uncertainty set, active coefficient indices, and
    RHS values are retained exactly.  Only the loading on each active support
    is changed from diagonal/identity to a dense square block.
    """
    if n_vars < 2:
        raise ValueError("n_vars must be at least 2.")

    params: Dict[str, Any] | None = None
    uncertainty_types: List[int] = []
    original_indices: List[List[int]] = []
    for _ in range(500):
        candidate = generate_random_backbone(n_vars)
        candidate_types = [
            int(candidate["objective_uncertainty_type"]),
            *[int(value) for value in candidate["constraint_types"]],
        ]
        candidate_indices = [
            [int(index) for index in indices]
            for indices in candidate["uncertainty_index"]
        ]
        uncertain_units = [
            unit_id
            for unit_id, type_code in enumerate(candidate_types)
            if type_code != DETERMINISTIC
        ]
        deterministic_units = [
            unit_id
            for unit_id, type_code in enumerate(candidate_types)
            if type_code == DETERMINISTIC
        ]
        if len(uncertain_units) != n_vars or len(deterministic_units) != 1:
            continue
        if len(candidate_indices) != n_vars + 1:
            continue

        valid_supports = True
        for unit_id in uncertain_units:
            support = candidate_indices[unit_id]
            nominal = np.asarray(_unit_nominal(candidate, unit_id), dtype=float)
            if (
                len(support) < 2
                or len(set(support)) != len(support)
                or any(index < 0 or index >= n_vars for index in support)
                or any(abs(float(nominal[index])) <= 1e-12 for index in support)
            ):
                valid_supports = False
                break
            unit_params = _unit_params(candidate, unit_id)
            if (
                candidate_types[unit_id] == BUDGET
                and float(unit_params["gamma"]) <= 0.0
            ):
                valid_supports = False
                break
        if not valid_supports:
            continue

        params = candidate
        uncertainty_types = candidate_types
        original_indices = candidate_indices
        break

    if params is None:
        raise RuntimeError(
            "Unable to sample a valid Random backbone for all-unit General P."
        )

    original_rhs = [
        float(row_params["b"])
        for row_params in params["constraint_params"]
    ]
    uncertain_units = [
        unit_id
        for unit_id, type_code in enumerate(uncertainty_types)
        if type_code != DETERMINISTIC
    ]
    original_uncertainty_data = [
        _backbone_uncertainty_snapshot(params, unit_id, type_code)
        for unit_id, type_code in enumerate(uncertainty_types)
    ]

    for unit_id in uncertain_units:
        type_code = uncertainty_types[unit_id]
        unit_params = _unit_params(params, unit_id)
        support = original_indices[unit_id]
        factor_dim = len(support)

        if type_code in (BOX, BUDGET):
            row_scales = [
                float(value) for value in unit_params["deviation"]
            ]
            scale_source = "original_random_deviation"
        else:
            row_scales = [1.0] * factor_dim
            scale_source = "original_random_identity"
        if len(row_scales) != factor_dim:
            raise AssertionError(
                "The Random deviation vector must match uncertainty_index."
            )

        active_block = _generate_dense_active_block(row_scales)
        ambient_loading = np.zeros((n_vars, factor_dim), dtype=float)
        ambient_loading[np.asarray(support, dtype=int), :] = active_block

        unit_params["factor_dim"] = factor_dim
        unit_params["factor_lower"] = (
            float(params["poly_lower"])
            if type_code == POLYHEDRAL
            else -1.0
        )
        unit_params["factor_upper"] = (
            float(params["poly_upper"])
            if type_code == POLYHEDRAL
            else 1.0
        )
        unit_params["loading_matrix"] = ambient_loading.tolist()
        unit_params["active_loading_matrix"] = active_block.tolist()
        unit_params["general_p_active_support"] = list(support)
        unit_params["general_p_row_scales"] = row_scales
        unit_params["general_p_scale_source"] = scale_source

    params["general_p_unit_ids"] = uncertain_units
    params["general_p_backbone_uncertainty_types"] = uncertainty_types.copy()
    params["general_p_original_uncertainty_index"] = original_indices
    params["general_p_original_uncertainty_data"] = (
        original_uncertainty_data
    )
    params["general_p_original_rhs"] = original_rhs
    return params


def _build_uncertainty_set(
    zeta: Any,
    uncertainty_type: int,
    uncertainty_params: Dict[str, Any],
) -> Any:
    if uncertainty_type == BOX:
        return rso.norm(zeta, np.inf) <= 1
    if uncertainty_type == BUDGET:
        return (
            rso.norm(zeta, np.inf) <= 1,
            rso.norm(zeta, 1) <= uncertainty_params["gamma"],
        )
    if uncertainty_type == POLYHEDRAL:
        conditions: List[Any] = []
        for coefficients, rhs, direction in uncertainty_params[
            "poly_constraints"
        ]:
            expression = sum(
                coefficients[index] * zeta[index]
                for index in range(len(coefficients))
            )
            if direction == 1:
                conditions.append(expression <= rhs)
            elif direction == -1:
                conditions.append(expression >= rhs)
            else:
                conditions.append(expression == rhs)
        lower = float(uncertainty_params.get("factor_lower", -1.0))
        upper = float(uncertainty_params.get("factor_upper", 1.0))
        for index in range(uncertainty_params["factor_dim"]):
            conditions.append(zeta[index] >= lower)
            conditions.append(zeta[index] <= upper)
        return tuple(conditions)
    raise ValueError(f"Unsupported uncertainty type: {uncertainty_type}")


def _loaded_coefficients(
    nominal_coefficients: Sequence[float],
    loading_matrix: Sequence[Sequence[float]],
    zeta: Any,
) -> List[Any]:
    factor_dim = len(loading_matrix[0])
    return [
        nominal_coefficients[row]
        + sum(
            loading_matrix[row][column] * zeta[column]
            for column in range(factor_dim)
        )
        for row in range(len(nominal_coefficients))
    ]


def solve_robust_optimization(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Solve a generated instance with RSOME to obtain ground truth."""
    global ro, rso
    from rsome import ro
    import rsome as rso

    n_vars = params["n_vars"]
    model = ro.Model()
    x = model.dvar(n_vars)

    for index in range(n_vars):
        model.st(x[index] >= params["x_lower"])
        model.st(x[index] <= params["x_upper"])

    objective_type = params["objective_uncertainty_type"]
    objective_coeffs = params["objective_coeffs"]
    if objective_type == DETERMINISTIC:
        objective_expression = sum(
            objective_coeffs[index] * x[index]
            for index in range(n_vars)
        )
        if params["objective_direction"]:
            model.max(objective_expression)
        else:
            model.min(objective_expression)
    else:
        objective_params = params["objective_uncertainty_params"]
        zeta_c = model.rvar(objective_params["factor_dim"])
        objective_set = _build_uncertainty_set(
            zeta_c,
            objective_type,
            objective_params,
        )
        actual_coefficients = _loaded_coefficients(
            objective_coeffs,
            objective_params["loading_matrix"],
            zeta_c,
        )
        objective_expression = sum(
            actual_coefficients[index] * x[index]
            for index in range(n_vars)
        )
        if params["objective_direction"]:
            model.maxmin(objective_expression, objective_set)
        else:
            model.minmax(objective_expression, objective_set)

    for row in range(params["n_constraints"]):
        constraint_type = params["constraint_types"][row]
        row_params = params["constraint_params"][row]
        if constraint_type == DETERMINISTIC:
            actual_coefficients = row_params["nominal_coeffs"]
            uncertainty_set = None
        else:
            zeta_i = model.rvar(row_params["factor_dim"])
            uncertainty_set = _build_uncertainty_set(
                zeta_i,
                constraint_type,
                row_params,
            )
            actual_coefficients = _loaded_coefficients(
                row_params["nominal_coeffs"],
                row_params["loading_matrix"],
                zeta_i,
            )

        lhs = sum(
            actual_coefficients[index] * x[index]
            for index in range(n_vars)
        )
        direction = row_params["direction"]
        if direction == 1:
            constraint = lhs <= row_params["b"]
        elif direction == -1:
            constraint = lhs >= row_params["b"]
        else:
            constraint = lhs == row_params["b"]

        if uncertainty_set is None:
            model.st(constraint)
        else:
            model.st(constraint.forall(uncertainty_set))

    try:
        model.solve()
        solution = np.asarray(x.get(), dtype=float)
        optimal_value = float(model.get())
    except Exception as exc:
        return {
            "status": "error",
            "solution": None,
            "optimal_value": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if (
        solution.shape != (n_vars,)
        or not np.all(np.isfinite(solution))
        or not np.isfinite(optimal_value)
    ):
        return {
            "status": "invalid",
            "solution": solution,
            "optimal_value": optimal_value,
        }

    return {
        "status": "optimal",
        "solution": solution,
        "optimal_value": optimal_value,
    }


def _polyhedral_standard_form(
    uncertainty_params: Dict[str, Any],
) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
    inequality_matrix: List[List[float]] = []
    inequality_rhs: List[float] = []
    equality_matrix: List[List[float]] = []
    equality_rhs: List[float] = []

    for coefficients, rhs, direction in uncertainty_params[
        "poly_constraints"
    ]:
        if direction == 1:
            inequality_matrix.append(list(coefficients))
            inequality_rhs.append(float(rhs))
        elif direction == -1:
            inequality_matrix.append(
                [-float(value) for value in coefficients]
            )
            inequality_rhs.append(-float(rhs))
        else:
            equality_matrix.append(list(coefficients))
            equality_rhs.append(float(rhs))

    return (
        inequality_matrix,
        inequality_rhs,
        equality_matrix,
        equality_rhs,
    )


def _row_symbols(row: int) -> Dict[str, str]:
    if row == 0:
        return {
            "nominal": "\\bar{\\mathbf{c}}",
            "loading": "\\mathbf{P}^{c}",
            "zeta": "\\boldsymbol{\\zeta}^{c}",
            "set": "\\mathcal{U}_{c}",
            "set_matrix": "\\mathbf{H}_{c}",
            "set_rhs": "\\mathbf{h}_{c}",
            "eq_matrix": "\\mathbf{E}_{c}",
            "eq_rhs": "\\mathbf{e}_{c}",
            "gamma": "\\Gamma_{c}",
        }
    return {
        "nominal": f"\\bar{{\\mathbf{{a}}}}_{{{row}}}",
        "loading": f"\\mathbf{{P}}_{{{row}}}",
        "zeta": f"\\boldsymbol{{\\zeta}}_{{{row}}}",
        "set": f"\\mathcal{{U}}_{{{row}}}",
        "set_matrix": f"\\mathbf{{H}}_{{{row}}}",
        "set_rhs": f"\\mathbf{{h}}_{{{row}}}",
        "eq_matrix": f"\\mathbf{{E}}_{{{row}}}",
        "eq_rhs": f"\\mathbf{{e}}_{{{row}}}",
        "gamma": f"\\Gamma_{{{row}}}",
    }


def _uncertainty_set_latex(
    row: int,
    uncertainty_type: int,
    uncertainty_params: Dict[str, Any],
) -> Tuple[str, List[str]]:
    symbols = _row_symbols(row)
    zeta = symbols["zeta"]
    set_name = symbols["set"]
    factor_dim = uncertainty_params["factor_dim"]
    definitions: List[str] = []

    if uncertainty_type == BOX:
        set_expression = (
            f"{set_name} = \\{{ {zeta} \\in \\mathbb{{R}}^{{{factor_dim}}}"
            f" : \\|{zeta}\\|_\\infty \\le 1 \\}}"
        )
    elif uncertainty_type == BUDGET:
        set_expression = (
            f"{set_name} = \\{{ {zeta} \\in \\mathbb{{R}}^{{{factor_dim}}}"
            f" : \\|{zeta}\\|_\\infty \\le 1,"
            f" \\|{zeta}\\|_1 \\le {symbols['gamma']} \\}}"
        )
        definitions.append(
            f"{symbols['gamma']} = {_fmt(uncertainty_params['gamma'])}"
        )
    elif uncertainty_type == POLYHEDRAL:
        (
            inequality_matrix,
            inequality_rhs,
            equality_matrix,
            equality_rhs,
        ) = _polyhedral_standard_form(uncertainty_params)
        conditions: List[str] = []
        if inequality_matrix:
            conditions.append(
                f"{symbols['set_matrix']} {zeta} \\le {symbols['set_rhs']}"
            )
            definitions.append(
                f"{symbols['set_matrix']} = {_mat(inequality_matrix)}"
            )
            definitions.append(
                f"{symbols['set_rhs']} = {_vec(inequality_rhs)}"
            )
        if equality_matrix:
            conditions.append(
                f"{symbols['eq_matrix']} {zeta} = {symbols['eq_rhs']}"
            )
            definitions.append(
                f"{symbols['eq_matrix']} = {_mat(equality_matrix)}"
            )
            definitions.append(
                f"{symbols['eq_rhs']} = {_vec(equality_rhs)}"
            )
        lower = float(uncertainty_params.get("factor_lower", -1.0))
        upper = float(uncertainty_params.get("factor_upper", 1.0))
        conditions.append(
            f"{_fmt(lower)} \\le {zeta}_k \\le {_fmt(upper)},"
            f"\\ k=1,\\ldots,{factor_dim}"
        )
        set_expression = (
            f"{set_name} = \\{{ {zeta} \\in \\mathbb{{R}}^{{{factor_dim}}}"
            f" : {', '.join(conditions)} \\}}"
        )
    else:
        raise ValueError(f"Unsupported uncertainty type: {uncertainty_type}")

    return set_expression, definitions


def generate_robust_description(params: Dict[str, Any]) -> str:
    """Render the single canonical compact matrix-style problem statement."""
    n_vars = params["n_vars"]
    lines = ["\\begin{align*}"]
    definitions: List[str] = []
    set_definitions: List[str] = []

    outer_operator = "\\max" if params["objective_direction"] else "\\min"
    objective_type = params["objective_uncertainty_type"]
    objective_symbols = _row_symbols(0)
    if objective_type == DETERMINISTIC:
        lines.append(
            f"{outer_operator}_{{\\mathbf{{x}}}} \\quad & "
            f"{objective_symbols['nominal']}^\\top \\mathbf{{x}}"
            " && \\text{[Deterministic Objective]} \\\\"
        )
    else:
        inner_operator = "\\min" if params["objective_direction"] else "\\max"
        lines.append(
            f"{outer_operator}_{{\\mathbf{{x}}}} \\quad & "
            f"{inner_operator}_{{{objective_symbols['zeta']} "
            f"\\in {objective_symbols['set']}}} "
            f"\\left({objective_symbols['nominal']} + "
            f"{objective_symbols['loading']} "
            f"{objective_symbols['zeta']}\\right)^\\top \\mathbf{{x}}"
            f" && \\text{{[{TYPE_LABELS[objective_type]} "
            "Support-Preserving General-P Objective]} \\\\"
        )
        objective_set, objective_set_values = _uncertainty_set_latex(
            0,
            objective_type,
            params["objective_uncertainty_params"],
        )
        set_definitions.append(objective_set)
        definitions.extend(objective_set_values)

    definitions.append(
        f"{objective_symbols['nominal']} = "
        f"{_vec(params['objective_coeffs'])}"
    )
    if objective_type > DETERMINISTIC:
        definitions.append(
            f"{objective_symbols['loading']} = "
            f"{_mat(params['objective_uncertainty_params']['loading_matrix'])}"
        )

    lines.append("& \\text{subject to} \\\\")
    for index in range(params["n_constraints"]):
        row = index + 1
        row_type = params["constraint_types"][index]
        row_params = params["constraint_params"][index]
        symbols = _row_symbols(row)
        relation = _ineq(row_params["direction"])

        if row_type == DETERMINISTIC:
            lines.append(
                f"& {symbols['nominal']}^\\top \\mathbf{{x}} "
                f"{relation} b_{{{row}}}"
                " && \\text{[Deterministic Constraint]} \\\\"
            )
        else:
            inner_operator = (
                "\\max" if row_params["direction"] == 1 else "\\min"
            )
            lines.append(
                f"& {inner_operator}_{{{symbols['zeta']} "
                f"\\in {symbols['set']}}} "
                f"\\left({symbols['nominal']} + "
                f"{symbols['loading']} {symbols['zeta']}\\right)^\\top "
                f"\\mathbf{{x}} {relation} b_{{{row}}}"
                f" && \\text{{[{TYPE_LABELS[row_type]} "
                "Support-Preserving General-P Constraint]} \\\\"
            )
            set_expression, set_values = _uncertainty_set_latex(
                row,
                row_type,
                row_params,
            )
            set_definitions.append(set_expression)
            definitions.extend(set_values)

        definitions.append(
            f"{symbols['nominal']} = "
            f"{_vec(row_params['nominal_coeffs'])}"
        )
        if row_type > DETERMINISTIC:
            definitions.append(
                f"{symbols['loading']} = "
                f"{_mat(row_params['loading_matrix'])}"
            )
        definitions.append(f"b_{{{row}}} = {_fmt(row_params['b'])}")

    lines.append(
        f"& {_fmt(params['x_lower'])} \\le x_j \\le "
        f"{_fmt(params['x_upper'])},"
        f" \\quad j=1,\\ldots,{n_vars} \\\\"
    )
    lines.append("& \\text{where} \\\\")
    for set_definition in set_definitions:
        lines.append(f"& {set_definition} \\\\")
    for definition in definitions:
        lines.append(f"& {definition} \\\\")
    lines.append("\\end{align*}")
    return "\n".join(lines)


def _is_non_boundary_solution(
    solution: np.ndarray,
    x_lower: float,
    x_upper: float,
) -> bool:
    on_a_bound = np.isclose(solution, x_lower) | np.isclose(
        solution,
        x_upper,
    )
    return not bool(np.all(on_a_bound))


def _general_p_metadata(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and expose the structural facts needed for dataset auditing."""
    uncertainty_types = [
        int(params["objective_uncertainty_type"]),
        *[int(value) for value in params["constraint_types"]],
    ]
    uncertain_rows = [
        row
        for row, uncertainty_type in enumerate(uncertainty_types)
        if uncertainty_type != DETERMINISTIC
    ]
    deterministic_rows = [
        row
        for row, uncertainty_type in enumerate(uncertainty_types)
        if uncertainty_type == DETERMINISTIC
    ]
    n_vars = int(params["n_vars"])
    if len(uncertain_rows) != n_vars or len(deterministic_rows) != 1:
        raise AssertionError(
            "The Random backbone must contain n uncertain units and exactly "
            "one deterministic unit."
        )
    backbone_types = [
        int(value)
        for value in params["general_p_backbone_uncertainty_types"]
    ]
    if backbone_types != uncertainty_types:
        raise AssertionError(
            "General-P conversion must not change any uncertainty family."
        )
    if [
        int(value) for value in params["general_p_unit_ids"]
    ] != uncertain_rows:
        raise AssertionError(
            "Every and only existing uncertain unit must receive General P."
        )

    original_indices = [
        [int(index) for index in indices]
        for indices in params["general_p_original_uncertainty_index"]
    ]
    current_indices = [
        [int(index) for index in indices]
        for indices in params["uncertainty_index"]
    ]
    if current_indices != original_indices or len(current_indices) != n_vars + 1:
        raise AssertionError(
            "General-P conversion must preserve every uncertainty_index."
        )

    unit_metadata: List[Dict[str, Any]] = []
    for unit_id in uncertain_rows:
        uncertainty_type = uncertainty_types[unit_id]
        uncertainty_params = _unit_params(params, unit_id)
        nominal = np.asarray(_unit_nominal(params, unit_id), dtype=float)
        support = current_indices[unit_id]
        factor_dim = int(uncertainty_params["factor_dim"])
        if factor_dim != len(support) or factor_dim < 2:
            raise AssertionError(
                "factor_dim must equal the original active-support size."
            )
        if (
            support != sorted(set(support))
            or any(index < 0 or index >= n_vars for index in support)
        ):
            raise AssertionError("The active support must be sorted and valid.")
        if any(abs(float(nominal[index])) <= 1e-12 for index in support):
            raise AssertionError(
                "General P may only perturb originally nonzero coefficients."
            )

        loading = np.asarray(
            uncertainty_params["loading_matrix"],
            dtype=float,
        )
        active_block = np.asarray(
            uncertainty_params["active_loading_matrix"],
            dtype=float,
        )
        if loading.shape != (n_vars, factor_dim):
            raise AssertionError(
                "Each ambient loading must be n x active factor_dim."
            )
        if active_block.shape != (factor_dim, factor_dim):
            raise AssertionError(
                "Each active General-P block must be square."
            )
        if not np.allclose(
            active_block,
            loading[np.asarray(support, dtype=int), :],
        ):
            raise AssertionError(
                "The active block was not embedded at uncertainty_index."
            )
        inactive_rows = [
            index for index in range(n_vars) if index not in set(support)
        ]
        if inactive_rows and np.count_nonzero(
            loading[np.asarray(inactive_rows, dtype=int), :]
        ):
            raise AssertionError(
                "A coefficient outside uncertainty_index received perturbation."
            )
        if np.count_nonzero(active_block) != active_block.size:
            raise AssertionError(
                "The active General-P block must be dense."
            )
        off_diagonal = active_block.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        if np.count_nonzero(off_diagonal) == 0:
            raise AssertionError(
                "The active General-P block must be non-diagonal."
            )
        if np.linalg.matrix_rank(active_block) != factor_dim:
            raise AssertionError(
                "The active General-P block must be full rank."
            )
        condition_number = float(np.linalg.cond(active_block))
        if not np.isfinite(condition_number) or condition_number > 50.0:
            raise AssertionError(
                "The active General-P block must be well conditioned."
            )

        row_scales = np.asarray(
            uncertainty_params["general_p_row_scales"],
            dtype=float,
        )
        if row_scales.shape != (factor_dim,):
            raise AssertionError("Invalid General-P row-scale metadata.")
        if not np.allclose(
            np.sum(np.abs(active_block), axis=1),
            row_scales,
            atol=0.005,
            rtol=0.0,
        ):
            raise AssertionError(
                "General-P active-row magnitudes drifted from Random scales."
            )
        if uncertainty_type in (BOX, BUDGET):
            original_deviation = np.asarray(
                uncertainty_params["deviation"],
                dtype=float,
            )
            if (
                uncertainty_params["general_p_scale_source"]
                != "original_random_deviation"
                or not np.array_equal(row_scales, original_deviation)
            ):
                raise AssertionError(
                    "Box/Budget row scales must equal original deviations."
                )
        elif (
            uncertainty_params["general_p_scale_source"]
            != "original_random_identity"
            or not np.array_equal(row_scales, np.ones(factor_dim))
        ):
            raise AssertionError(
                "Polyhedral active rows must use the original identity scale."
            )

        set_params: Dict[str, Any] = {}
        if uncertainty_type == BOX:
            set_params["deviation"] = uncertainty_params["deviation"]
        elif uncertainty_type == BUDGET:
            set_params["deviation"] = uncertainty_params["deviation"]
            set_params["gamma"] = float(uncertainty_params["gamma"])
            if float(uncertainty_params["gamma"]) <= 0.0:
                raise AssertionError("Budget uncertainty must be non-vacuous.")
        elif uncertainty_type == POLYHEDRAL:
            for coefficients, _, _ in uncertainty_params["poly_constraints"]:
                if len(coefficients) != factor_dim:
                    raise AssertionError(
                        "Polyhedral constraints must stay in active coordinates."
                    )
            set_params["poly_constraints"] = uncertainty_params[
                "poly_constraints"
            ]
            set_params["factor_bounds"] = [
                float(uncertainty_params["factor_lower"]),
                float(uncertainty_params["factor_upper"]),
            ]

        backbone_uncertainty_data = params[
            "general_p_original_uncertainty_data"
        ][unit_id]
        current_uncertainty_data = _backbone_uncertainty_snapshot(
            params,
            unit_id,
            uncertainty_type,
        )
        if current_uncertainty_data != backbone_uncertainty_data:
            raise AssertionError(
                "General-P conversion changed a family or uncertainty set."
            )

        unit_label = (
            "objective" if unit_id == 0 else f"constraint_{unit_id}"
        )
        unit_metadata.append(
            {
                "unit_id": int(unit_id),
                "unit_label": unit_label,
                "row_kind": (
                    "objective" if unit_id == 0 else "constraint"
                ),
                "uncertainty_family": TYPE_NAMES[uncertainty_type],
                "active_support": list(support),
                "factor_dim": factor_dim,
                "ambient_loading_shape": [n_vars, factor_dim],
                "loading_matrix": uncertainty_params["loading_matrix"],
                "active_loading_matrix": uncertainty_params[
                    "active_loading_matrix"
                ],
                "active_block_is_dense": True,
                "active_block_is_non_diagonal": True,
                "active_block_is_full_rank": True,
                "active_block_condition_number": condition_number,
                "zero_outside_active_support": True,
                "active_rows_have_nonzero_nominal_coefficients": True,
                "mixing_normalization": (
                    "positive_near_identity_row_l1_normalized_before_rounding"
                ),
                "row_scale_source": uncertainty_params[
                    "general_p_scale_source"
                ],
                "row_scales": uncertainty_params["general_p_row_scales"],
                "backbone_uncertainty_data": backbone_uncertainty_data,
                "uncertainty_set_params": set_params,
            }
        )

    for constraint_index, row_params in enumerate(params["constraint_params"]):
        if int(row_params["direction"]) == 0 and uncertainty_types[
            constraint_index + 1
        ] != DETERMINISTIC:
            raise AssertionError("Equality constraints must remain deterministic.")
        if bool(row_params.get("b_uncertain", False)):
            raise AssertionError("RHS uncertainty must remain disabled.")
        if "b_deviation" in row_params:
            raise AssertionError("RHS deviation data must remain absent.")

    current_rhs = [
        float(row_params["b"])
        for row_params in params["constraint_params"]
    ]
    if current_rhs != [
        float(value) for value in params["general_p_original_rhs"]
    ]:
        raise AssertionError("General-P conversion changed an RHS value.")

    return _json_safe({
        "base_generator": "synthetic.generate_random_params",
        "row_indexing": "0=objective, 1..n=constraint rows",
        "coupling_mode": "independent_units",
        "general_p_scope": "all_existing_uncertain_units",
        "support_mode": "original_random_uncertainty_index",
        "all_uncertain_units_use_general_p": True,
        "support_preserving": True,
        "ambient_loading_is_sparse_by_design": True,
        "general_p_unit_ids": uncertain_rows,
        "general_p_unit_labels": [
            "objective" if unit_id == 0 else f"constraint_{unit_id}"
            for unit_id in uncertain_rows
        ],
        "general_p_units": unit_metadata,
        "num_general_p_units": len(unit_metadata),
        "random_backbone_uncertain_unit_count": len(uncertain_rows),
        "random_backbone_deterministic_unit_count": len(deterministic_rows),
        "deterministic_unit": int(deterministic_rows[0]),
        "backbone_uncertainty_types": backbone_types,
        "original_uncertainty_index": original_indices,
        "backbone_unit_uncertainty_data": params[
            "general_p_original_uncertainty_data"
        ],
        "rhs_uncertainty": False,
        "template": "T111",
    })


def batch_generate_json(
    var_list: Sequence[int],
    samples_per_n: int,
    seed: Optional[int] = None,
    max_attempts_per_n: int = 10000,
) -> List[Dict[str, Any]]:
    """Generate accepted instances and return JSON-serializable records."""
    if samples_per_n <= 0:
        raise ValueError("samples_per_n must be positive.")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    records: List[Dict[str, Any]] = []
    for n_vars in var_list:
        attempt_count = 0
        with tqdm(
            total=samples_per_n,
            desc=f"general-P {n_vars}d",
        ) as progress:
            for slot_index in range(1, samples_per_n + 1):
                while True:
                    attempt_count += 1
                    if attempt_count > max_attempts_per_n:
                        raise RuntimeError(
                            f"Failed to obtain {samples_per_n} accepted "
                            f"instances for n={n_vars} after "
                            f"{max_attempts_per_n} attempts."
                        )

                    params = generate_random_params(n_vars)
                    generation_metadata = _general_p_metadata(params)
                    result = solve_robust_optimization(params)
                    if result["status"] != "optimal":
                        continue

                    solution = np.asarray(result["solution"], dtype=float)
                    if not _is_non_boundary_solution(
                        solution,
                        params["x_lower"],
                        params["x_upper"],
                    ):
                        continue

                    record_id = f"{n_vars}_{slot_index}_GP"
                    uncertainty_types = [
                        params["objective_uncertainty_type"],
                        *params["constraint_types"],
                    ]
                    records.append(
                        {
                            "id": record_id,
                            "n_vars": n_vars,
                            "n_constraints": params["n_constraints"],
                            "template_param": CANONICAL_TEMPLATE_PARAM.copy(),
                            "setting": SETTING_NAME,
                            "description": generate_robust_description(params),
                            "uncertainty_type": uncertainty_types,
                            "general_p_unit_ids": generation_metadata[
                                "general_p_unit_ids"
                            ],
                            "general_p_unit_labels": generation_metadata[
                                "general_p_unit_labels"
                            ],
                            "row_indexing": generation_metadata["row_indexing"],
                            "support_mode": generation_metadata["support_mode"],
                            "solution": solution.tolist(),
                            "optimal_value": result["optimal_value"],
                            "generation_metadata": generation_metadata,
                        }
                    )
                    progress.update(1)
                    break

        print(
            f"general-P {n_vars}d: accepted {samples_per_n} "
            f"after {attempt_count} attempts"
        )

    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate robust LP data where all existing uncertain units use "
            "support-preserving general perturbation matrices."
        )
    )
    parser.add_argument(
        "--vars",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5],
        help="Decision-variable dimensions (default: 2 3 4 5).",
    )
    parser.add_argument(
        "--samples-per-n",
        type=int,
        default=8,
        help="Accepted instances per dimension (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260726,
        help="Random seed (default: 20260726).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets")
        / "test_general_p_32.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = batch_generate_json(
        var_list=args.vars,
        samples_per_n=args.samples_per_n,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} records to {args.output.resolve()}")


if __name__ == "__main__":
    main()
