"""Numerical models and renderers for the literature-sourced external test set."""
from __future__ import annotations
import itertools
import math
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def quiet_model(name: str) -> gp.Model:
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    model.Params.NumericFocus = 2
    return model

def solve_portfolio(kind: str, parameter: float) -> tuple[float, list[float]]:
    n = 199
    mu = [1.05 + 0.3 * (200 - ell) / 199 for ell in range(1, 200)]
    sig = [0.05 + 0.6 * (200 - ell) / 199 for ell in range(1, 200)]
    m = quiet_model(f'portfolio_{kind}_{parameter}')
    y = m.addVars(200, lb=0.0, name='y')
    t = m.addVar(lb=-GRB.INFINITY, name='t')
    m.addConstr(gp.quicksum((y[i] for i in range(200))) == 1)
    nominal = gp.quicksum((mu[i] * y[i] for i in range(n))) + 1.05 * y[199]
    a = [sig[i] * y[i] for i in range(n)]
    if kind == 'box':
        support = gp.quicksum(a)
    elif kind == 'budget':
        pi = m.addVar(lb=0.0, name='pi')
        q = m.addVars(n, lb=0.0, name='q')
        m.addConstrs((q[i] >= a[i] - pi for i in range(n)))
        support = gp.quicksum((q[i] for i in range(n))) + parameter * pi
    else:
        raise ValueError(kind)
    m.addConstr(t <= nominal - support)
    m.setObjective(t, GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f'Portfolio solve failed: {m.Status}')
    sol = [y[i].X for i in range(200)] + [t.X]
    return (m.ObjVal, sol)

def portfolio_latex(kind: str, parameter: float) -> str:
    if kind == 'box':
        uset = '\\|\\boldsymbol\\zeta\\|_\\infty\\le1'
    elif kind == 'budget':
        uset = f'\\|\\boldsymbol\\zeta\\|_\\infty\\le1,\\ \\|\\boldsymbol\\zeta\\|_1\\le {parameter:g}'
    else:
        uset = f'\\|\\boldsymbol\\zeta\\|_\\infty\\le1,\\ \\|\\boldsymbol\\zeta\\|_2\\le {parameter:g}'
    return f'\\begin{{align*}}\n\\max_{{\\boldsymbol y,t}}\\quad&t\\\\\n\\text{{s.t.}}\\quad\n&\\sum_{{\\ell=1}}^{{199}}(\\mu_\\ell+\\sigma_\\ell\\zeta_\\ell)y_\\ell\n+1.05y_{{200}}\\ge t,\\quad\\forall\\boldsymbol\\zeta\\in\\mathcal U,\\\\\n&\\sum_{{\\ell=1}}^{{200}}y_\\ell=1,\\qquad y_\\ell\\ge0,\\\\\n&\\mu_\\ell=1.05+0.3\\frac{{200-\\ell}}{{199}},\\quad\n\\sigma_\\ell=0.05+0.6\\frac{{200-\\ell}}{{199}},\\quad \\ell=1,\\ldots,199,\\\\\n&\\mathcal U=\\{{\\boldsymbol\\zeta\\in\\mathbb R^{{199}}:{uset}\\}}.\n\\end{{align*}}'

def solve_lf2012(kind: str, parameter: float) -> tuple[float, list[float]]:
    P = [180, 180, 250, 270, 300, 320]
    C = [20, 25, 30, 40, 50, 60]
    V = [2] * 6
    U = [1500, 2000, 2200, 3000, 2700, 2500]
    D = [1100, 1500, 1800, 1600, 2300, 2500]
    m = quiet_model(f'lf2012_{kind}')
    x = m.addVars(6, lb=0.0, name='x')
    y = m.addVars(6, lb=0.0, name='y')
    z = m.addVars(6, lb=0.0, name='z')
    m.addConstr(500 + x[0] - y[0] - z[0] == 0)
    m.addConstrs((y[j - 1] + x[j] - y[j] - z[j] == 0 for j in range(1, 6)))
    m.addConstr(y[5] == 500)
    m.addConstrs((x[j] <= U[j] for j in range(6)))
    m.addConstrs((z[j] <= D[j] for j in range(6)))
    nominal = gp.quicksum((C[j] * x[j] + V[j] * y[j] for j in range(6)))
    a = [0.5 * C[j] * x[j] for j in range(6)]
    if kind == 'polyhedral':
        eta = m.addVar(lb=0.0, name='eta')
        m.addConstrs((eta >= a[j] for j in range(6)))
        support = parameter * eta
    elif kind == 'interval_polyhedral':
        pi = m.addVar(lb=0.0, name='pi')
        q = m.addVars(6, lb=0.0, name='q')
        m.addConstrs((q[j] >= a[j] - pi for j in range(6)))
        support = gp.quicksum((q[j] for j in range(6))) + parameter * pi
    else:
        raise ValueError(kind)
    m.addConstr(nominal + support <= 400000)
    m.setObjective(gp.quicksum((P[j] * z[j] for j in range(6))), GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError((kind, m.Status))
    solution = [x[j].X for j in range(6)] + [y[j].X for j in range(6)] + [z[j].X for j in range(6)]
    return (m.ObjVal, solution)

def lf2012_latex(kind: str, parameter: float) -> str:
    if kind == 'polyhedral':
        uset = f'\\|\\boldsymbol\\xi\\|_1\\le{parameter:g}'
    else:
        uset = f'\\|\\boldsymbol\\xi\\|_\\infty\\le1,\\ \\|\\boldsymbol\\xi\\|_1\\le{parameter:g}'
    return f'\\begin{{align*}}\n\\max_{{\\boldsymbol x,\\boldsymbol y,\\boldsymbol z}}\\quad&\\sum_{{j=1}}^6P_jz_j\\\\\n\\text{{s.t.}}\\quad\n&\\sum_{{j=1}}^6 C_j(1+0.5\\xi_j)x_j+\\sum_{{j=1}}^6V_jy_j\\le400000,\n\\quad\\forall\\boldsymbol\\xi\\in\\mathcal U,\\\\\n&500+x_1-y_1-z_1=0,\\\\\n&y_{{j-1}}+x_j-y_j-z_j=0,\\quad j=2,\\ldots,6,\\\\\n&y_6=500,\\quad 0\\le x_j\\le U_j,\\quad0\\le z_j\\le D_j,\\quad y_j\\ge0,\\\\\n&\\boldsymbol P=(180,180,250,270,300,320),\\\\\n&\\boldsymbol C=(20,25,30,40,50,60),\\quad\\boldsymbol V=(2,2,2,2,2,2),\\\\\n&\\boldsymbol U=(1500,2000,2200,3000,2700,2500),\\\\\n&\\boldsymbol D=(1100,1500,1800,1600,2300,2500),\\\\\n&\\mathcal U=\\{{\\boldsymbol\\xi\\in\\mathbb R^6:{uset}\\}}.\n\\end{{align*}}'

def support(model: gp.Model, expressions: list[Any], kind: str, name: str, radius: float=1.0) -> Any:
    """Support of nonnegative expressions over the paper's unit set."""
    if kind == 'box':
        return radius * gp.quicksum(expressions)
    if kind == 'polyhedral':
        eta = model.addVar(lb=0.0, name=f'eta_{name}')
        for index, expr in enumerate(expressions):
            model.addConstr(eta >= expr, name=f'support_{name}_{index}')
        return radius * eta
    raise ValueError(kind)

def uncertainty_set_latex(kind: str, radius: float=1.0) -> str:
    if kind == 'box':
        return f'\\mathcal U_{{\\infty}}=\\{{\\boldsymbol\\xi:\\|\\boldsymbol\\xi\\|_\\infty\\le{radius:g}\\}}'
    return f'\\mathcal U_{{1}}=\\{{\\boldsymbol\\xi:\\|\\boldsymbol\\xi\\|_1\\le{radius:g}\\}}'

def solve_ldf_compact_lp(case: str, kind: str, radius: float=1.0) -> tuple[float, list[float]]:
    model = quiet_model(f'ldf_lp_{case}_{kind}')
    x1 = model.addVar(lb=0.0, name='x1')
    x2 = model.addVar(lb=0.0, name='x2')
    row1_dev = [x1, 2 * x2]
    row2_dev = [0.6 * x1, 0.8 * x2]
    if case == 'all':
        row1_dev.append(14.0)
        row2_dev.append(7.2)
    model.addConstr(10 * x1 + 20 * x2 + support(model, row1_dev, kind, 'r1', radius) <= 140)
    model.addConstr(6 * x1 + 8 * x2 + support(model, row2_dev, kind, 'r2', radius) <= 72)
    objective = 8 * x1 + 12 * x2
    if case == 'all':
        objective -= support(model, [0.8 * x1, 1.2 * x2], kind, 'obj', radius)
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError((case, kind, model.Status))
    return (model.ObjVal, [x1.X, x2.X])

def ldf_lp_latex(case: str, kind: str, radius: float=1.0) -> str:
    uset = uncertainty_set_latex(kind, radius)
    if case == 'lhs':
        objective = '\\max_{x_1,x_2}\\quad 8x_1+12x_2'
        rhs1, rhs2 = ('140', '72')
        quantifiers = '\\forall\\boldsymbol\\xi_1,\\boldsymbol\\xi_2\\in\\mathcal U\\ \\text{(independent by row)}'
    else:
        objective = '\\max_{x_1,x_2}\\ \\min_{\\boldsymbol\\xi_0\\in\\mathcal U}(8+0.8\\xi_{01})x_1+(12+1.2\\xi_{02})x_2'
        rhs1, rhs2 = ('140+14\\xi_{13}', '72+7.2\\xi_{23}')
        quantifiers = '\\forall\\boldsymbol\\xi_0,\\boldsymbol\\xi_1,\\boldsymbol\\xi_2\\in\\mathcal U\\ \\text{(independent by row)}'
    return f"\\begin{{align*}}\n{objective}\\\\\n\\text{{s.t.}}\\quad\n&(10+\\xi_{{11}})x_1+(20+2\\xi_{{12}})x_2\\le {rhs1},\\\\\n&(6+0.6\\xi_{{21}})x_1+(8+0.8\\xi_{{22}})x_2\\le {rhs2},\\\\\n&x_1,x_2\\ge0,\\qquad {quantifiers},\\\\\n&\\mathcal U={uset.split('=', 1)[1]}.\n\\end{{align*}}"

def solve_ldf_compact_milp(case: str, kind: str) -> tuple[float, list[float]]:
    model = quiet_model(f'ldf_milp_{case}_{kind}')
    x1 = model.addVar(lb=0.0, ub=10.0, name='x1')
    x2 = model.addVar(lb=0.0, ub=10.0, name='x2')
    y1 = model.addVar(vtype=GRB.BINARY, name='y1')
    y2 = model.addVar(vtype=GRB.BINARY, name='y2')
    rows = [(x1 + x2, 20.0, [0.1 * x1, 0.1 * x2]), (x1 + 2 * x2, 12.0, [0.1 * x1, 0.2 * x2]), (x1 - 20 * y1, 0.0, [0.1 * x1, 2 * y1]), (x2 - 20 * y2, 0.0, [0.1 * x2, 2 * y2]), (x1 - x2, -4.0, [0.1 * x1, 0.1 * x2])]
    for ridx, (nominal, rhs, deviations) in enumerate(rows, start=1):
        protection = support(model, deviations, kind, f'r{ridx}') if case == 'lhs' else 0
        model.addConstr(nominal + protection <= rhs)
    objective = 3 * x1 + 2 * x2 - 10 * y1 - 5 * y2
    if case == 'obj':
        objective -= support(model, [0.3 * x1, 0.2 * x2, y1, 0.5 * y2], kind, 'obj')
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError((case, kind, model.Status))
    return (model.ObjVal, [x1.X, x2.X, y1.X, y2.X])

def ldf_milp_latex(case: str, kind: str) -> str:
    uset = uncertainty_set_latex(kind).split('=', 1)[1]
    matrix = 'A=\\begin{bmatrix}1&1&0&0\\\\1&2&0&0\\\\1&0&-20&0\\\\0&1&0&-20\\\\1&-1&0&0\\end{bmatrix},\\quad\\ b=(20,12,0,0,-4)^\\top'
    if case == 'lhs':
        objective = '\\max\\quad3x_1+2x_2-10y_1-5y_2'
        constraints = '\\widetilde a_i^\\top(x_1,x_2,y_1,y_2)\\le b_i,\\quad\\widetilde a_{ij}=a_{ij}+0.1|a_{ij}|\\xi_{ij},\\quad\\forall\\boldsymbol\\xi_i\\in\\mathcal U,\\ i=1,\\ldots,5'
    else:
        objective = '\\max\\ \\min_{\\boldsymbol\\xi_0\\in\\mathcal U}(3+0.3\\xi_{01})x_1+(2+0.2\\xi_{02})x_2-(10+\\xi_{03})y_1-(5+0.5\\xi_{04})y_2'
        constraints = 'A(x_1,x_2,y_1,y_2)^\\top\\le b'
    return f'\\begin{{align*}}\n&{objective}\\\\\n\\text{{s.t.}}\\quad&{constraints},\\\\\n&{matrix},\\\\\n&0\\le x_1,x_2\\le10,\\quad y_1,y_2\\in\\{{0,1\\}},\\\\\n&\\mathcal U={uset}.\n\\end{{align*}}'

def solve_vertex_lp(objective: list[float], objective_constant: float, inequalities: list[tuple[list[float], float]]) -> tuple[float, list[float]]:
    """Solve a small bounded LP by exhaustive vertex enumeration."""
    n_vars = len(objective)
    matrix = np.asarray([row for row, _ in inequalities], dtype=float)
    rhs = np.asarray([bound for _, bound in inequalities], dtype=float)
    best_value = -math.inf
    best_point: np.ndarray | None = None
    for active in itertools.combinations(range(len(inequalities)), n_vars):
        active_matrix = matrix[list(active)]
        if np.linalg.matrix_rank(active_matrix) < n_vars:
            continue
        try:
            point = np.linalg.solve(active_matrix, rhs[list(active)])
        except np.linalg.LinAlgError:
            continue
        if np.max(matrix @ point - rhs) > 1e-08:
            continue
        value = float(np.dot(np.asarray(objective), point) + objective_constant)
        if value > best_value + 1e-09:
            best_value = value
            best_point = point
    if best_point is None:
        raise RuntimeError('LP has no enumerated feasible vertex')
    return (best_value, [float(value) for value in best_point])

def support_patterns(terms: list[tuple[list[float], float]], kind: str, radius: float) -> list[tuple[list[float], float]]:
    """Affine pieces whose maximum is the uncertainty support function."""
    if kind == 'box':
        coeff = [sum((term[0][index] for term in terms)) * radius for index in range(len(terms[0][0]))]
        const = sum((term[1] for term in terms)) * radius
        return [(coeff, const)]
    if kind == 'polyhedral':
        return [([radius * value for value in coeff], radius * const) for coeff, const in terms]
    if kind == 'box_polyhedral':
        full = int(math.floor(radius + 1e-12))
        fraction = radius - full
        if full >= len(terms):
            coeff = [sum((term[0][index] for term in terms)) for index in range(len(terms[0][0]))]
            return [(coeff, sum((term[1] for term in terms)))]
        patterns: list[tuple[list[float], float]] = []
        for chosen in itertools.combinations(range(len(terms)), full):
            remaining = [index for index in range(len(terms)) if index not in chosen]
            next_indices = remaining if fraction > 1e-12 else [None]
            for next_index in next_indices:
                coeff = [sum((terms[index][0][column] for index in chosen)) for column in range(len(terms[0][0]))]
                const = sum((terms[index][1] for index in chosen))
                if next_index is not None:
                    coeff = [coeff[column] + fraction * terms[next_index][0][column] for column in range(len(coeff))]
                    const += fraction * terms[next_index][1]
                patterns.append((coeff, const))
        return patterns
    raise ValueError(kind)

def add_robust_row(inequalities: list[tuple[list[float], float]], nominal: tuple[list[float], float], rhs: float, terms: list[tuple[list[float], float]], kind: str, radius: float) -> None:
    nominal_coeff, nominal_const = nominal
    for support_coeff, support_const in support_patterns(terms, kind, radius):
        inequalities.append(([nominal_coeff[index] + support_coeff[index] for index in range(len(nominal_coeff))], rhs - nominal_const - support_const))

def solve_ldf_rows_lp(case: str, kind: str, radius: float) -> tuple[float, list[float]]:
    objective_terms = [([0.8, 0.0], 0.0), ([0.0, 1.2], 0.0)]
    row_data = [([10.0, 20.0], 140.0, [([1.0, 0.0], 0.0), ([0.0, 2.0], 0.0)], 14.0), ([6.0, 8.0], 72.0, [([0.6, 0.0], 0.0), ([0.0, 0.8], 0.0)], 7.2)]
    use_obj = case == 'all' and True
    n_vars = 3 if use_obj else 2
    inequalities: list[tuple[list[float], float]] = []
    for row_index, (coeff, rhs, lhs_terms, rhs_dev) in enumerate(row_data, start=1):
        terms: list[tuple[list[float], float]] = []
        terms.extend((([*term_coeff, *([0.0] if n_vars == 3 else [])], const) for term_coeff, const in lhs_terms))
        terms.append(([0.0] * n_vars, rhs_dev))
        nominal_coeff = [*coeff, *([0.0] if n_vars == 3 else [])]
        if terms:
            add_robust_row(inequalities, (nominal_coeff, 0.0), rhs, terms, kind, radius)
        else:
            inequalities.append((nominal_coeff, rhs))
    inequalities.extend([([-1.0, 0.0] + ([0.0] if n_vars == 3 else []), 0.0), ([0.0, -1.0] + ([0.0] if n_vars == 3 else []), 0.0)])
    if use_obj:
        inequalities.append(([0.0, 0.0, -1.0], 0.0))
        padded_terms = [([*coeff, 0.0], const) for coeff, const in objective_terms]
        for coeff, const in support_patterns(padded_terms, kind, radius):
            inequalities.append(([coeff[0], coeff[1], -1.0], -const))
        objective = [8.0, 12.0, -1.0]
    else:
        objective = [8.0, 12.0]
    value, point = solve_vertex_lp(objective, 0.0, inequalities)
    return (value, point[:2])
MILP_ROWS = [([1.0, 1.0], 0.0, 20.0, [([0.1, 0.0], 0.0), ([0.0, 0.1], 0.0)], 2.0), ([1.0, 2.0], 0.0, 12.0, [([0.1, 0.0], 0.0), ([0.0, 0.2], 0.0)], 1.2), ([1.0, 0.0], -20.0, 0.0, [([0.1, 0.0], 0.0)], 0.0), ([0.0, 1.0], -20.0, 0.0, [([0.0, 0.1], 0.0)], 0.0), ([1.0, -1.0], 0.0, -4.0, [([0.1, 0.0], 0.0), ([0.0, 0.1], 0.0)], 0.4)]

def row_uncertainty_set_latex(kind: str, radius: float, dimension: str='m') -> str:
    if kind == 'box':
        return f'\\{{\\boldsymbol\\xi\\in\\mathbb R^{{{dimension}}}:|\\xi_j|\\le {radius:g}\\}}'
    if kind == 'polyhedral':
        return f'\\{{\\boldsymbol\\xi\\in\\mathbb R^{{{dimension}}}:\\sum_j|\\xi_j|\\le {radius:g}\\}}'
    if kind == 'box_polyhedral':
        return f'\\{{\\boldsymbol\\xi\\in\\mathbb R^{{{dimension}}}:|\\xi_j|\\le1,\\ \\sum_j|\\xi_j|\\le {radius:g}\\}}'
    raise ValueError(kind)

def style_bits(style: str) -> tuple[bool, bool, bool]:
    if len(style) != 4 or style[0] != 'T':
        raise ValueError(style)
    return (style[1] == '1', style[2] == '1', style[3] == '1')

def label(text: str, named: bool) -> str:
    return f'\\quad\\text{{[{text}]}}' if named else ''

def serialize_ldf_lp(style: str, case: str, kind: str, radius: float) -> str:
    inner, compact, named = style_bits(style)
    family_name = {'box': 'Box', 'polyhedral': 'Polyhedral', 'box_polyhedral': 'Box+Polyhedral'}[kind]
    objective_tag = label(f'{family_name} Uncertainty Objective', named and case == 'all')
    row_tag = label(f'{family_name} Uncertainty Constraint', named)
    obj_set = f'\\mathcal U_{{0,{family_name}}}^{{(2)}}'
    row1_set = f'\\mathcal U_{{1,{family_name}}}^{{(3)}}'
    row2_set = f'\\mathcal U_{{2,{family_name}}}^{{(3)}}'
    if compact:
        obj_expr = '(\\bar{\\mathbf c}+\\mathbf d^c\\odot\\boldsymbol\\xi_0)^\\top\\mathbf x'
        row1_expr = '(\\bar{\\mathbf a}_1+\\mathbf d_1^a\\odot\\boldsymbol\\xi_{1,1:2})^\\top\\mathbf x-14\\xi_{13}'
        row2_expr = '(\\bar{\\mathbf a}_2+\\mathbf d_2^a\\odot\\boldsymbol\\xi_{2,1:2})^\\top\\mathbf x-7.2\\xi_{23}'
        objective_data = '\\bar{\\mathbf c}=(8,12)^\\top,\\ \\mathbf d^c=(0.8,1.2)^\\top,\\ ' if case == 'all' else ''
        data = f'\\mathbf x=(x_1,x_2)^\\top,\\ {objective_data}\\\\\n&\\bar{{\\mathbf a}}_1=(10,20)^\\top,\\ \\mathbf d_1^a=(1,2)^\\top,\\ \\bar{{\\mathbf a}}_2=(6,8)^\\top,\\ \\mathbf d_2^a=(0.6,0.8)^\\top,\\\\\n&'
    else:
        obj_expr = '(8+0.8\\xi_{01})x_1+(12+1.2\\xi_{02})x_2'
        row1_expr = '(10+\\xi_{11})x_1+(20+2\\xi_{12})x_2-14\\xi_{13}'
        row2_expr = '(6+0.6\\xi_{21})x_1+(8+0.8\\xi_{22})x_2-7.2\\xi_{23}'
        data = ''
    if case == 'all':
        if inner:
            objective = f'\\max_{{x_1,x_2}}\\ \\min_{{\\boldsymbol\\xi_0\\in{obj_set}}}\\{{{obj_expr}\\}}{objective_tag}'
        else:
            objective = f'\\max_{{x_1,x_2}}\\quad {obj_expr},\\quad\\forall\\boldsymbol\\xi_0\\in{obj_set}{objective_tag}'
    else:
        objective = '\\max_{x_1,x_2}\\quad8x_1+12x_2'
    if inner:
        row1 = f'\\max_{{\\boldsymbol\\xi_1\\in{row1_set}}}\\{{{row1_expr}\\}}\\le140'
        row2 = f'\\max_{{\\boldsymbol\\xi_2\\in{row2_set}}}\\{{{row2_expr}\\}}\\le72'
    else:
        row1 = f'{row1_expr}\\le140,\\quad\\forall\\boldsymbol\\xi_1\\in{row1_set}'
        row2 = f'{row2_expr}\\le72,\\quad\\forall\\boldsymbol\\xi_2\\in{row2_set}'
    set2 = row_uncertainty_set_latex(kind, radius, '2')
    set3 = row_uncertainty_set_latex(kind, radius, '3')
    if case == 'all':
        set_lines = f'&{data}\\mathcal U_{{i,{family_name}}}^{{(3)}}={set3}\\quad(i=1,2),\\\\\n&\\mathcal U_{{0,{family_name}}}^{{(2)}}={set2}.'
    else:
        set_lines = f'&{data}\\mathcal U_{{i,{family_name}}}^{{(3)}}={set3}\\quad(i=1,2).'
    return f'\\begin{{align*}}\n&{objective}\\\\\n\\text{{s.t.}}\\quad&{row1}{row_tag},\\\\\n&{row2}{row_tag},\\\\\n&x_1,x_2\\ge0,\\\\\n{set_lines}\n\\end{{align*}}'

def solve_ldf_rows_milp(case: str, kind: str, radius: float) -> tuple[float, list[float]]:
    """Exact MILP solution by four binary assignments and LP vertices."""
    use_objective = case == 'all' and True
    best_value = -math.inf
    best_solution: list[float] = []
    for y1, y2 in itertools.product((0.0, 1.0), repeat=2):
        n_continuous = 3 if use_objective else 2
        pad = [0.0] if use_objective else []
        inequalities: list[tuple[list[float], float]] = [([-1.0, 0.0, *pad], 0.0), ([0.0, -1.0, *pad], 0.0), ([1.0, 0.0, *pad], 10.0), ([0.0, 1.0, *pad], 10.0)]
        if use_objective:
            inequalities.append(([0.0, 0.0, -1.0], 0.0))
        for row_index, (coeff, y_coeff, rhs, lhs_terms, rhs_dev) in enumerate(MILP_ROWS):
            y_value = y1 if row_index == 2 else y2 if row_index == 3 else 0.0
            terms: list[tuple[list[float], float]] = []
            terms.extend((([term[0][0], term[0][1], *pad], term[1]) for term in lhs_terms))
            if row_index == 2:
                terms.append(([0.0] * n_continuous, 2.0 * y1))
            if row_index == 3:
                terms.append(([0.0] * n_continuous, 2.0 * y2))
            if True and rhs_dev:
                terms.append(([0.0] * n_continuous, rhs_dev))
            nominal = ([coeff[0], coeff[1], *pad], y_coeff * y_value)
            if terms:
                add_robust_row(inequalities, nominal, rhs, terms, kind, radius)
            else:
                inequalities.append((nominal[0], rhs - nominal[1]))
        if use_objective:
            objective_terms = [([0.3, 0.0, 0.0], 0.0), ([0.0, 0.2, 0.0], 0.0), ([0.0, 0.0, 0.0], y1), ([0.0, 0.0, 0.0], 0.5 * y2)]
            for coefficient, constant in support_patterns(objective_terms, kind, radius):
                inequalities.append(([coefficient[0], coefficient[1], -1.0], -constant))
            objective = [3.0, 2.0, -1.0]
        else:
            objective = [3.0, 2.0]
        try:
            value, point = solve_vertex_lp(objective, -10.0 * y1 - 5.0 * y2, inequalities)
        except RuntimeError:
            continue
        if value > best_value + 1e-09:
            best_value = value
            best_solution = [point[0], point[1], y1, y2]
    if not best_solution:
        raise RuntimeError(f'No feasible LDF MILP assignment: {case}, {kind}, {radius}')
    return (best_value, best_solution)

def serialize_ldf_milp(case: str, kind: str, radius: float) -> str:
    family = {'box': 'Box', 'polyhedral': 'Polyhedral', 'box_polyhedral': 'BoxPolyhedral'}[kind]
    row_data = [('(1+0.1\\xi_{11})x_1+(1+0.1\\xi_{12})x_2-2\\xi_{13}', '20', 3), ('(1+0.1\\xi_{21})x_1+(2+0.2\\xi_{22})x_2-1.2\\xi_{23}', '12', 3), ('(1+0.1\\xi_{31})x_1+(-20+2\\xi_{32})y_1', '0', 2), ('(1+0.1\\xi_{41})x_2+(-20+2\\xi_{42})y_2', '0', 2), ('(1+0.1\\xi_{51})x_1+(-1+0.1\\xi_{52})x_2-0.4\\xi_{53}', '-4', 3)]
    rows = []
    for index, (expression, rhs, dimension) in enumerate(row_data, start=1):
        set_name = f'\\mathcal U_{{{index},{family}}}^{{({dimension})}}'
        rows.append(f'\\max_{{\\boldsymbol\\xi_{index}\\in{set_name}}}\\{{{expression}\\}}\\le{rhs}')
    if case == 'all':
        objective = f'\\max_{{x_1,x_2,y_1,y_2}}\\ \\min_{{\\boldsymbol\\xi_0\\in\\mathcal U_{{0,{family}}}^{{(4)}}}}\\{{(3+0.3\\xi_{{01}})x_1+(2+0.2\\xi_{{02}})x_2-(10+\\xi_{{03}})y_1-(5+0.5\\xi_{{04}})y_2\\}}'
    else:
        objective = '\\max_{x_1,x_2,y_1,y_2}\\quad3x_1+2x_2-10y_1-5y_2'
    set2 = row_uncertainty_set_latex(kind, radius, '2')
    set3 = row_uncertainty_set_latex(kind, radius, '3')
    set_lines = [f'\\mathcal U_{{i,{family}}}^{{(2)}}={set2}\\quad(i=3,4)', f'\\mathcal U_{{i,{family}}}^{{(3)}}={set3}\\quad(i=1,2,5)']
    if case == 'all':
        set4 = row_uncertainty_set_latex(kind, radius, '4')
        set_lines.append(f'\\mathcal U_{{0,{family}}}^{{(4)}}={set4}')
    return '\\begin{align*}\n' + '&' + objective + '\\\\\n\\text{s.t.}\\quad&' + ',\\\\\n&'.join(rows) + ',\\\\\n&0\\le x_1,x_2\\le10,\\quad y_1,y_2\\in\\{0,1\\},\\\\\n&' + ',\\\\\n&'.join(set_lines) + '.\n\\end{align*}'

def parse_dat(path: Path) -> tuple[int, int, int, list[int], list[int]]:
    lines = [line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines()]
    if len(lines) != 3:
        raise ValueError(f'Expected exactly three lines in {path}, found {len(lines)}')

    def parse_line(line: str) -> list[int]:
        return [int(item) for item in line.split(';') if item.strip()]
    header = parse_line(lines[0])
    if len(header) != 3:
        raise ValueError(f'Malformed n;p;Gamma header in {path}: {lines[0]!r}')
    n, p, gamma = header
    nominal = parse_line(lines[1])
    deviations = parse_line(lines[2])
    if len(nominal) != n or len(deviations) != n:
        raise ValueError(f'Coefficient count mismatch in {path}: n={n}, len(c)={len(nominal)}, len(d)={len(deviations)}')
    if not 0 <= gamma <= p <= n:
        raise ValueError(f'Invalid n/p/Gamma values in {path}: {header}')
    if any((value < 0 for value in deviations)):
        raise ValueError(f'Negative deviation found in {path}')
    return (n, p, gamma, nominal, deviations)

def robust_value(nominal: list[int], deviations: list[int], selected: Iterable[int], gamma: int) -> int:
    selected_list = list(selected)
    worst_deviations = sorted((deviations[index] for index in selected_list), reverse=True)[:gamma]
    return sum((nominal[index] for index in selected_list)) + sum(worst_deviations)

def solve_selection(nominal: list[int], deviations: list[int], p: int, gamma: int) -> tuple[int, list[int], dict[str, object]]:
    """Solve the integer-Gamma robust selection problem exactly.

    For a fixed Bertsimas--Sim threshold pi, the p items with smallest
    c_i + max(d_i-pi, 0) are optimal.  Some global optimum has
    pi in {0} union {d_i}, so enumerating these thresholds is exact.
    """
    candidates: dict[tuple[int, ...], int] = {}
    best_counterpart = None
    best_counterpart_selected: tuple[int, ...] | None = None
    best_pi: int | None = None
    for pi in sorted({0, *deviations}):
        adjusted = [nominal[index] + max(deviations[index] - pi, 0) for index in range(len(nominal))]
        selected = tuple(sorted(range(len(nominal)), key=lambda index: (adjusted[index], index))[:p])
        counterpart_value = gamma * pi + sum((adjusted[index] for index in selected))
        candidates[selected] = robust_value(nominal, deviations, selected, gamma)
        counterpart_key = (counterpart_value, selected, pi)
        incumbent_key = (best_counterpart if best_counterpart is not None else float('inf'), best_counterpart_selected or tuple(), best_pi if best_pi is not None else 0)
        if counterpart_key < incumbent_key:
            best_counterpart = counterpart_value
            best_counterpart_selected = selected
            best_pi = pi
    selected, objective = min(candidates.items(), key=lambda item: (item[1], item[0]))
    if best_counterpart != objective:
        raise AssertionError(f'Threshold/direct verification mismatch: {best_counterpart} != {objective}')
    solution = [1 if index in selected else 0 for index in range(len(nominal))]
    if sum(solution) != p:
        raise AssertionError('Selection solution does not contain exactly p items')
    verification = {'method': 'exact Bertsimas-Sim threshold enumeration', 'thresholds_checked': len({0, *deviations}), 'best_pi': best_pi, 'counterpart_objective': best_counterpart, 'direct_worst_case_objective': robust_value(nominal, deviations, selected, gamma), 'selected_indices_1_based': [index + 1 for index in selected]}
    return (objective, solution, verification)

def latex_vector(values: Iterable[int | float]) -> str:
    return ', '.join((str(value) for value in values))

def selection_latex(nominal: list[int], deviations: list[int], n: int, p: int, gamma: int) -> str:
    template = '\\begin{align*}\n\\min_{\\boldsymbol{x}}\\quad\n& \\max_{\\boldsymbol{\\zeta}\\in\\mathcal{U}_{__GAMMA__}}\n  \\sum_{i=1}^{__N__}\\left(c_i+d_i\\zeta_i\\right)x_i \\\\\n\\text{s.t.}\\quad\n& \\sum_{i=1}^{__N__}x_i=__P__, \\\\\n& x_i\\in\\{0,1\\},\\quad i=1,\\ldots,__N__, \\\\\n& \\mathcal{U}_{__GAMMA__}=\\left\\{\\boldsymbol{\\zeta}\\in\\mathbb{R}^{__N__}:\n0\\leq\\zeta_i\\leq 1\\ (i=1,\\ldots,__N__),\\ \n\\sum_{i=1}^{__N__}\\zeta_i\\leq __GAMMA__\\right\\}, \\\\\n& (c_1,\\ldots,c_{__N__})=(__NOMINAL__), \\\\\n& (d_1,\\ldots,d_{__N__})=(__DEVIATIONS__).\n\\end{align*}'
    return template.replace('__GAMMA__', str(gamma)).replace('__N__', str(n)).replace('__P__', str(p)).replace('__NOMINAL__', latex_vector(nominal)).replace('__DEVIATIONS__', latex_vector(deviations))

DRUG_DESCRIPTION = '\\begin{align*}\n\\min_{R_1,R_2,D_1,D_2}\\quad\n&100R_1+199.9R_2-5500D_1-6100D_2\\\\\n\\text{s.t.}\\quad\n&(0.01+0.00005\\xi_1)R_1+(0.02+0.0004\\xi_2)R_2\n-0.5D_1-0.6D_2\\ge0,\\quad\\forall\\boldsymbol\\xi\\in\\mathcal U,\\\\\n&R_1+R_2\\le1000,\\quad 90D_1+100D_2\\le2000,\\\\\n&40D_1+50D_2\\le800,\\\\\n&100R_1+199.9R_2+700D_1+800D_2\\le100000,\\\\\n&R_1,R_2,D_1,D_2\\ge0,\\\\\n&\\mathcal U=\\{\\boldsymbol\\xi\\in\\mathbb R^2:-1\\le\\xi_1,\\xi_2\\le1\\}.\n\\end{align*}'
