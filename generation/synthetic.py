"""Generate random robust linear programs for AutoREM training."""

import numpy as np
import random
import json
from tqdm import tqdm


def _rand_nonzero_uniform(low, high, size=None, cutoff=0.1):
    """Generate uniform random numbers but push them away from zero."""
    val = np.random.uniform(low, high, size)

    if size is None:
        if abs(val) < cutoff:
            val = cutoff if val >= 0 else -cutoff
        return val

    val = np.where(np.abs(val) < cutoff, np.sign(val) * cutoff, val)

    val = np.where(val == 0.0, cutoff, val)
    return val


def _x0_poly_sample(poly_lower, poly_upper, poly_range, n):
    """Generate interior feasibility point for polyhedral uncertainty set (1 decimal)."""
    if poly_lower < 0 and poly_upper > 0:
        return np.round(np.random.uniform(poly_lower, poly_upper, size=n), 1)
    elif poly_lower == 0.0:
        return np.round(np.random.uniform(0.0, poly_upper, size=n), 1)
    else:
        return np.round(np.random.uniform(poly_lower, 0.0, size=n), 1)


def generate_feasible_lp(n, x0, base_range=(-10, 10)):
    """Generate a bounded linear program with x0 as a feasible point."""

    unmask_idx = []

    m = n


    A = np.round(_rand_nonzero_uniform(base_range[0], base_range[1], size=(m, n)), 1)
    if n > 2:
        mask = np.ones((m, n))
        for i in range(m):
            mask[i] = np.random.randint(0, 2, size=n)
            while mask[i].sum() < 2:
                mask[i] = np.random.randint(0, 2, size=n)
            nonzero_idx = np.where(mask[i] != 0)[0]
            unmask_idx.append(nonzero_idx.tolist())
        A = A * mask


    directions = np.random.choice([1, -1], size=m)


    if n >= 3 and np.random.rand() > 0.5:
        eq_idx = np.random.randint(0, m)
        directions[eq_idx] = 0


    b = np.zeros(m)
    for i in range(m):
        ax = np.round(np.dot(A[i], x0), 2)
        if directions[i] == 1:
            b[i] = np.round(np.random.uniform(ax, np.sum(np.abs(A[i]) * base_range[1])), 2)
        elif directions[i] == -1:
            b[i] = np.round(np.random.uniform(-np.sum(np.abs(A[i]) * base_range[1]), ax), 2)
        else:
            b[i] = ax

    return A, b, directions, unmask_idx


def generate_random_params(n_vars):
    """Sample a robust LP with exactly one deterministic objective or constraint row."""

    objective_coeffs = [round(random.uniform(-10, 10), 1) for _ in range(n_vars)]


    objective_direction = np.random.choice([True, False])


    nominal_range = round(random.uniform(1, 10), 1)

    x_bound_type = random.choice(['a', 'b', 'c'])
    if x_bound_type == 'a':
        x_lower, x_upper = -nominal_range, nominal_range
        x0 = np.round(np.random.uniform(-nominal_range, nominal_range, size=n_vars), 1)
    elif x_bound_type == 'b':
        x_lower, x_upper = 0, nominal_range
        x0 = np.round(np.random.uniform(0.0, nominal_range, size=n_vars), 1)
    else:
        x_lower, x_upper = -nominal_range, 0
        x0 = -np.round(np.random.uniform(0.0, nominal_range, size=n_vars), 1)
    nominal_A_poly, nominal_b_poly, nominal_directions, unmask_idx = generate_feasible_lp(n=n_vars, x0=x0, base_range=(
        -nominal_range, nominal_range))

    uncertainty_index = []


    n_constraints = n_vars


    eq_indices = [j for j in range(n_constraints) if nominal_directions[j] == 0]
    has_equality = len(eq_indices) > 0

    if has_equality:

        deterministic_row = eq_indices[0] + 1
    else:

        deterministic_row = random.randint(0, n_constraints)


    if deterministic_row == 0:
        objective_uncertainty_type = 0
    else:
        objective_uncertainty_type = random.randint(1, 3)

    uncertain_idx = []

    objective_uncertainty_params = {}
    poly_range = round(0.1 * nominal_range, 1)

    poly_bound_type = random.choice(['a', 'b', 'c'])
    if poly_bound_type == 'a':
        poly_lower, poly_upper = -poly_range, poly_range
    elif poly_bound_type == 'b':
        poly_lower, poly_upper = 0.0, poly_range
    else:
        poly_lower, poly_upper = -poly_range, 0.0

    if objective_uncertainty_type > 0:
        if n_vars > 2:
            n_uncer = np.random.randint(2, n_vars + 1)
            uncertain_idx = np.sort(np.random.choice(range(n_vars), size=n_uncer, replace=False)).tolist()
        else:
            n_uncer = 2
            uncertain_idx = [0, 1]

        if objective_uncertainty_type == 1:
            objective_uncertainty_params['deviation'] = [
                np.round(max(0.1, np.random.uniform(0.05, 0.2) * abs(objective_coeffs[uncertain_idx[i]])), 1) for i in
                range(n_uncer)
            ]

        elif objective_uncertainty_type == 2:
            objective_uncertainty_params['deviation'] = [
                np.round(max(0.1, np.random.uniform(0.05, 0.2) * abs(objective_coeffs[uncertain_idx[i]])), 1) for i in
                range(n_uncer)
            ]


            objective_uncertainty_params['gamma'] = round(random.uniform(0, n_uncer), 1)

        elif objective_uncertainty_type == 3:

            x0_poly = _x0_poly_sample(poly_lower, poly_upper, poly_range, n_uncer)
            A_poly, b_poly, directions, _ = generate_feasible_lp(n=n_uncer, x0=x0_poly,
                                                                 base_range=(-poly_range, poly_range))

            poly_constraints = []
            num_poly_constraints = n_uncer - 1

            for i in range(num_poly_constraints):
                poly_constraints.append((A_poly[i].tolist(), b_poly[i], directions[i]))
            objective_uncertainty_params['poly_constraints'] = poly_constraints

    uncertainty_index.append(uncertain_idx)


    constraint_types = []
    constraint_params = []


    for j in range(n_constraints):
        uncertain_idx = []

        if j + 1 == deterministic_row or nominal_directions[j] == 0:

            constraint_type = 0
        else:
            constraint_type = random.randint(1, 3)
        constraint_types.append(constraint_type)


        params = {}


        params['nominal_coeffs'] = nominal_A_poly[j].tolist()

        params['b'] = nominal_b_poly[j]

        params['direction'] = nominal_directions[j]

        if constraint_type > 0:
            if n_vars > 2:
                n_uncer = np.random.randint(2, len(unmask_idx[j]) + 1)
                uncertain_idx = np.sort(np.random.choice(unmask_idx[j], size=n_uncer, replace=False)).tolist()
            else:
                n_uncer = 2
                uncertain_idx = [0, 1]

            if constraint_type == 1:
                params['deviation'] = [
                    np.round(max(0.1, np.random.uniform(0.05, 0.2) * abs(params['nominal_coeffs'][uncertain_idx[i]])),
                             1) for i in range(n_uncer)
                ]

            elif constraint_type == 2:
                params['deviation'] = [
                    np.round(max(0.1, np.random.uniform(0.05, 0.2) * abs(params['nominal_coeffs'][uncertain_idx[i]])),
                             1) for i in range(n_uncer)
                ]


                params['gamma'] = round(random.uniform(0, n_uncer), 1)

            elif constraint_type == 3:

                x0_poly = _x0_poly_sample(poly_lower, poly_upper, poly_range, n_uncer)
                A_poly, b_poly, directions, _ = generate_feasible_lp(n=n_uncer, x0=x0_poly,
                                                                     base_range=(-poly_range, poly_range))

                poly_constraints = []
                num_poly_constraints = n_uncer - 1

                for i in range(num_poly_constraints):
                    poly_constraints.append((A_poly[i].tolist(), b_poly[i], directions[i]))
                params['poly_constraints'] = poly_constraints

        uncertainty_index.append(uncertain_idx)

        constraint_params.append(params)

    return {
        'x_0': x0,
        'n_vars': n_vars,
        'n_constraints': n_constraints,
        'objective_coeffs': objective_coeffs,
        'objective_direction': objective_direction,
        'objective_uncertainty_type': objective_uncertainty_type,
        'objective_uncertainty_params': objective_uncertainty_params,
        'constraint_types': constraint_types,
        'constraint_params': constraint_params,
        'uncertainty_index': uncertainty_index,
        'nominal_range': nominal_range,
        'poly_range': poly_range,
        'x_lower': x_lower,
        'x_upper': x_upper,
        'poly_lower': poly_lower,
        'poly_upper': poly_upper,
    }


def solve_robust_optimization(params):
    """Solve a robust LP with RSOME and return the optimum, solution, and status."""
    from rsome import ro
    import rsome as rso

    n_vars = params['n_vars']
    n_constraints = params['n_constraints']
    objective_coeffs = params['objective_coeffs']
    objective_direction = params['objective_direction']
    objective_uncertainty_type = params['objective_uncertainty_type']
    objective_uncertainty_params = params['objective_uncertainty_params']
    constraint_types = params['constraint_types']
    constraint_params = params['constraint_params']
    objective_uncertainty_index = params['uncertainty_index'][0]
    constraint_uncertainty_index = params['uncertainty_index'][1:]
    nominal_range = params['nominal_range']
    poly_range = params['poly_range']
    x_lower = params['x_lower']
    x_upper = params['x_upper']
    poly_lower = params['poly_lower']
    poly_upper = params['poly_upper']


    model = ro.Model()


    x = model.dvar(n_vars)
    for i in range(n_vars):
        model.st(x[i] >= x_lower)
        model.st(x[i] <= x_upper)


    if objective_uncertainty_type == 0:
        obj_expr = sum(objective_coeffs[i] * x[i] for i in range(n_vars))
        if objective_direction:
            model.max(obj_expr)
        else:
            model.min(obj_expr)

    else:

        z_obj = model.rvar(len(objective_uncertainty_index))

        if objective_uncertainty_type == 1:

            obj_deviation = objective_uncertainty_params['deviation']


            obj_uncertainty_set = (rso.norm(z_obj, np.inf) <= 1)


            obj_coeffs_actual = objective_coeffs.copy()
            for i in range(len(objective_uncertainty_index)):
                obj_coeffs_actual[objective_uncertainty_index[i]] = objective_coeffs[objective_uncertainty_index[i]] + \
                                                                    obj_deviation[i] * z_obj[i]


            obj_expr = sum(obj_coeffs_actual[i] * x[i] for i in range(n_vars))


            if objective_direction:
                model.maxmin(obj_expr, obj_uncertainty_set)
            else:
                model.minmax(obj_expr, obj_uncertainty_set)

        elif objective_uncertainty_type == 2:

            obj_deviation = objective_uncertainty_params['deviation']
            gamma = objective_uncertainty_params['gamma']


            obj_uncertainty_set = (
                rso.norm(z_obj, np.inf) <= 1,
                rso.norm(z_obj, 1) <= gamma
            )


            obj_coeffs_actual = objective_coeffs.copy()
            for i in range(len(objective_uncertainty_index)):
                obj_coeffs_actual[objective_uncertainty_index[i]] = objective_coeffs[objective_uncertainty_index[i]] + \
                                                                    obj_deviation[i] * z_obj[i]


            obj_expr = sum(obj_coeffs_actual[i] * x[i] for i in range(n_vars))


            if objective_direction:
                model.maxmin(obj_expr, obj_uncertainty_set)
            else:
                model.minmax(obj_expr, obj_uncertainty_set)

        elif objective_uncertainty_type == 3:

            poly_constraints = objective_uncertainty_params['poly_constraints']


            obj_uncertainty_conditions = []
            for poly_coeffs, poly_rhs, poly_direction in poly_constraints:
                val_expr = sum(poly_coeffs[i] * z_obj[i] for i in range(len(objective_uncertainty_index)))
                if poly_direction == 1:
                    condition = val_expr <= poly_rhs
                elif poly_direction == -1:
                    condition = val_expr >= poly_rhs
                else:
                    condition = val_expr == poly_rhs
                obj_uncertainty_conditions.append(condition)
            for i in range(len(objective_uncertainty_index)):
                obj_uncertainty_conditions.append(z_obj[i] >= poly_lower)
                obj_uncertainty_conditions.append(z_obj[i] <= poly_upper)


            obj_uncertainty_set = tuple(obj_uncertainty_conditions)


            obj_coeffs_actual = objective_coeffs.copy()
            for i in range(len(objective_uncertainty_index)):
                obj_coeffs_actual[objective_uncertainty_index[i]] += z_obj[i]


            obj_expr = sum(obj_coeffs_actual[i] * x[i] for i in range(n_vars))


            if objective_direction:
                model.maxmin(obj_expr, obj_uncertainty_set)
            else:
                model.minmax(obj_expr, obj_uncertainty_set)


    for i in range(n_constraints):
        constraint_type = constraint_types[i]
        params = constraint_params[i]


        A_nominal = params['nominal_coeffs']
        b = params['b']
        direction = params['direction']
        uncertainty_idx = constraint_uncertainty_index[i]

        if constraint_type == 0:

            constr_expr = sum(A_nominal[j] * x[j] for j in range(n_vars))
            if direction == 1:
                model.st(constr_expr <= b)
            elif direction == -1:
                model.st(constr_expr >= b)
            else:
                model.st(constr_expr == b)

        elif constraint_type > 0:

            z = model.rvar(len(uncertainty_idx))
            if constraint_type == 1:

                A_deviation = params['deviation']


                uncertainty_set = (rso.norm(z, np.inf) <= 1)


                A_actual = A_nominal.copy()
                for j in range(len(uncertainty_idx)):
                    A_actual[uncertainty_idx[j]] = A_nominal[uncertainty_idx[j]] + A_deviation[j] * z[j]


            elif constraint_type == 2:

                A_deviation = params['deviation']
                gamma = params['gamma']


                uncertainty_set = (
                    rso.norm(z, np.inf) <= 1,
                    rso.norm(z, 1) <= gamma
                )


                A_actual = A_nominal.copy()
                for j in range(len(uncertainty_idx)):
                    A_actual[uncertainty_idx[j]] = A_nominal[uncertainty_idx[j]] + A_deviation[j] * z[j]


            elif constraint_type == 3:

                poly_constraints = params['poly_constraints']


                uncertainty_conditions = []
                for poly_coeffs, poly_rhs, poly_direction in poly_constraints:
                    val_expr = sum(poly_coeffs[j] * z[j] for j in range(len(uncertainty_idx)))
                    if poly_direction == 1:
                        condition = val_expr <= poly_rhs
                    elif poly_direction == -1:
                        condition = val_expr >= poly_rhs
                    else:
                        condition = val_expr == poly_rhs
                    uncertainty_conditions.append(condition)
                for j in range(len(uncertainty_idx)):
                    uncertainty_conditions.append(z[j] >= poly_lower)
                    uncertainty_conditions.append(z[j] <= poly_upper)


                uncertainty_set = tuple(uncertainty_conditions)


                A_actual = A_nominal.copy()
                for j in range(len(uncertainty_idx)):
                    A_actual[uncertainty_idx[j]] += z[j]


            constr_expr = sum(A_actual[j] * x[j] for j in range(n_vars))
            if direction == 1:
                constraint = constr_expr <= b
            elif direction == -1:
                constraint = constr_expr >= b
            else:
                constraint = constr_expr == b

            model.st(constraint.forall(uncertainty_set))

    robust_attempts = 0
    try:
        robust_attempts += 1

        model.solve()


        solution = x.get()
        optimal_value = model.get()


        return {
            'status': 'optimal',
            'solution': solution,
            'optimal_value': optimal_value,
            'external_success': True,
            'robust_attempts': robust_attempts,
        }

    except Exception as e:

        error_msg = str(e)

        if ("unbounded" in error_msg.lower()) or ("Scipy solution status: 3" in error_msg):
            return {
                'status': 'unbounded',

                'external_success': False,
                'robust_attempts': robust_attempts
            }
        elif ("infeasible" in error_msg.lower()) or (
                "Scipy solution status: 2" in error_msg):
            return {
                'status': 'infeasible',

                'external_success': False,
                'robust_attempts': robust_attempts
            }
        else:
            return {
                'status': 'error',

                'external_success': False,
                'robust_attempts': robust_attempts
            }


def _fmt(v):
    if abs(v) < 1e-10:
        return "0"
    if v == int(v):
        return str(int(v))
    return f"{v:.4f}".rstrip('0').rstrip('.')


def _fmt_p(v):
    """Format value, wrapping negatives in parentheses for use after - or + signs."""
    s = _fmt(v)
    if v < 0:
        return f"({s})"
    return s


def _ineq(direction):
    if direction == 1 or direction is True:
        return "\\le"
    elif direction == -1 or direction is False:
        return "\\ge"
    return "="


def _nominal_expr(n_vars, coeffs):
    """Build scalar expr with all actual values: 4.00x_{1} - 2.50x_{2}"""
    terms = []
    first = True
    for j in range(n_vars):
        c = coeffs[j]
        if abs(c) < 1e-10:
            continue
        if first and c >= 0:
            terms.append(f"{_fmt(abs(c))}x_{{{j + 1}}}")
        else:
            terms.append(f"{'+ ' if c >= 0 else '- '}{_fmt(abs(c))}x_{{{j + 1}}}")
        first = False
    return " ".join(terms) if terms else "0"


def _extract_params(params):
    """Extract common parameters from params dict."""
    return (params['n_vars'], params['n_constraints'], params['objective_coeffs'],
            params['objective_direction'], params['objective_uncertainty_type'],
            params['objective_uncertainty_params'], params['constraint_types'],
            params['constraint_params'], params['uncertainty_index'][0],
            params['uncertainty_index'][1:], params['nominal_range'], params['poly_range'],
            params['x_lower'], params['x_upper'], params['poly_lower'], params['poly_upper'])


def _x_bnd_str(j, xl, xu):
    """LaTeX bound string for a single decision variable x_{j+1}."""
    if xl == 0:
        return f"0 \\le x_{{{j + 1}}} \\le {_fmt(xu)}"
    elif xu == 0:
        return f"{_fmt(xl)} \\le x_{{{j + 1}}} \\le 0"
    else:
        return f"{_fmt(xl)} \\le x_{{{j + 1}}} \\le {_fmt(xu)}"


def _x_bnd_compact(xl, xu, nv):
    """Compact ∀j bound string; works because bounds are per-instance (all vars same)."""
    if xl == 0:
        return f"0 \\le x_j \\le {_fmt(xu)}, \\quad \\forall j \\in \\{{1, \\dots, {nv}\\}}"
    elif xu == 0:
        return f"{_fmt(xl)} \\le x_j \\le 0, \\quad \\forall j \\in \\{{1, \\dots, {nv}\\}}"
    else:
        return f"{_fmt(xl)} \\le x_j \\le {_fmt(xu)}, \\quad \\forall j \\in \\{{1, \\dots, {nv}\\}}"


def _poly_bnd_str(vn, pl, pu):
    """LaTeX bound string for a polyhedral uncertainty variable."""
    if pl == 0.0:
        return f"0 \\le {vn} \\le {_fmt(pu)}"
    elif pu == 0.0:
        return f"{_fmt(pl)} \\le {vn} \\le 0"
    else:
        return f"{_fmt(pl)} \\le {vn} \\le {_fmt(pu)}"


def _poly_set_deviation_scalar(vnames, poly_constraints, pl, pu):
    parts = []
    for coeffs, rhs, direction in poly_constraints:
        ct = []
        first = True
        for k in range(len(vnames)):
            c = coeffs[k]
            if abs(c) < 1e-10: continue
            if first and c >= 0:
                ct.append(f"{_fmt(abs(c))}{vnames[k]}")
            else:
                ct.append(f"{'+ ' if c >= 0 else '- '}{_fmt(abs(c))}{vnames[k]}")
            first = False
        parts.append(f"{' '.join(ct)} {_ineq(direction)} {_fmt(rhs)}")
    for vn in vnames:
        parts.append(_poly_bnd_str(vn, pl, pu))
    return parts


def _template_000(params):
    nv, nc, oc, od, out, oup, cts, cps, oui, cui, nr, pr, xl, xu, pl, pu = _extract_params(params)
    L = ["\\begin{align*}"]
    opt = "\\max" if od else "\\min"


    if out == 0:
        L.append(f"{opt} \\quad & {_nominal_expr(nv, oc)} \\\\")
    else:
        terms = []
        first = True
        for j in range(nv):
            c = oc[j]
            if j in oui:
                if out == 3:
                    sym = f"({_fmt(c)} + \\zeta^c_{{{j + 1}}})"
                else:
                    sym = f"c_{{{j + 1}}}"

                if first:
                    terms.append(f"{sym}x_{{{j + 1}}}")
                else:
                    terms.append(f"+ {sym}x_{{{j + 1}}}")
            else:
                if abs(c) < 1e-10:
                    continue
                if first and c >= 0:
                    terms.append(f"{_fmt(abs(c))}x_{{{j + 1}}}")
                else:
                    terms.append(f"{'+ ' if c >= 0 else '- '}{_fmt(abs(c))}x_{{{j + 1}}}")
            first = False

        expr = " ".join(terms).replace("+ -", "- ")
        if out == 3:
            vl = "(" + ", ".join([f"\\zeta^c_{{{j + 1}}}" for j in oui]) + ")"
        else:
            vl = "(" + ", ".join([f"c_{{{j + 1}}}" for j in oui]) + ")"
        L.append(f"{opt} \\quad & {expr}, \\quad \\forall {vl} \\in \\mathcal{{U}}_{{c}} \\\\")


    for i in range(nc):
        ct, cp, ui = cts[i], cps[i], cui[i]
        ci = i + 1
        iq = _ineq(cp['direction'])

        if ct == 0:
            L.append(f"& {_nominal_expr(nv, cp['nominal_coeffs'])} {iq} {_fmt(cp['b'])} \\\\")
        else:
            terms = []
            first = True
            for j in range(nv):
                c = cp['nominal_coeffs'][j]
                if j in ui:
                    if ct == 3:
                        sym = f"({_fmt(c)} + \\zeta^{{a{ci}}}_{{{j + 1}}})"
                    else:
                        sym = f"a_{{{ci}{j + 1}}}"

                    if first:
                        terms.append(f"{sym}x_{{{j + 1}}}")
                    else:
                        terms.append(f"+ {sym}x_{{{j + 1}}}")
                else:
                    if abs(c) < 1e-10:
                        continue
                    if first and c >= 0:
                        terms.append(f"{_fmt(abs(c))}x_{{{j + 1}}}")
                    else:
                        terms.append(f"{'+ ' if c >= 0 else '- '}{_fmt(abs(c))}x_{{{j + 1}}}")
                first = False

            expr = " ".join(terms).replace("+ -", "- ")
            if ct == 3:
                vl = "(" + ", ".join([f"\\zeta^{{a{ci}}}_{{{j + 1}}}" for j in ui]) + ")"
            else:
                vl = "(" + ", ".join([f"a_{{{ci}{j + 1}}}" for j in ui]) + ")"
            L.append(f"& {expr} {iq} {_fmt(cp['b'])}, \\quad \\forall {vl} \\in \\mathcal{{U}}_{{a{ci}}} \\\\")


    bnd = ", \\quad ".join([_x_bnd_str(j, xl, xu) for j in range(nv)])
    L.append(f"& {bnd} \\\\")


    def _set000_obj():
        if out == 1:
            vl = "(" + ", ".join([f"c_{{{j + 1}}}" for j in oui]) + ")"
            sp = [f"|c_{{{j + 1}}} - {_fmt_p(oc[j])}| \\le {_fmt(oup['deviation'][k])}" for k, j in enumerate(oui)]
        elif out == 2:
            vl = "(" + ", ".join([f"c_{{{j + 1}}}" for j in oui]) + ")"
            sp = [f"|c_{{{j + 1}}} - {_fmt_p(oc[j])}| \\le {_fmt(oup['deviation'][k])}" for k, j in enumerate(oui)]
            sp.append(" + ".join(
                [f"\\frac{{|c_{{{j + 1}}} - {_fmt_p(oc[j])}|}}{{{_fmt(oup['deviation'][k])}}}" for k, j in
                 enumerate(oui)]) + f" \\le {_fmt(oup['gamma'])}")
        elif out == 3:
            vnames = [f"\\zeta^c_{{{j + 1}}}" for j in oui]
            vl = "(" + ", ".join(vnames) + ")"
            sp = _poly_set_deviation_scalar(vnames, oup['poly_constraints'], pl, pu)
        return vl, sp

    def _set000_con(ci, ct, cp, ui):
        if ct == 1:
            vl = "(" + ", ".join([f"a_{{{ci}{j + 1}}}" for j in ui]) + ")"
            sp = [f"|a_{{{ci}{j + 1}}} - {_fmt_p(cp['nominal_coeffs'][j])}| \\le {_fmt(cp['deviation'][k])}" for k, j in
                  enumerate(ui)]
        elif ct == 2:
            vl = "(" + ", ".join([f"a_{{{ci}{j + 1}}}" for j in ui]) + ")"
            sp = [f"|a_{{{ci}{j + 1}}} - {_fmt_p(cp['nominal_coeffs'][j])}| \\le {_fmt(cp['deviation'][k])}" for k, j in
                  enumerate(ui)]
            sp.append(" + ".join(
                [f"\\frac{{|a_{{{ci}{j + 1}}} - {_fmt_p(cp['nominal_coeffs'][j])}|}}{{{_fmt(cp['deviation'][k])}}}" for
                 k, j in enumerate(ui)]) + f" \\le {_fmt(cp['gamma'])}")
        elif ct == 3:
            vnames = [f"\\zeta^{{a{ci}}}_{{{j + 1}}}" for j in ui]
            vl = "(" + ", ".join(vnames) + ")"
            sp = _poly_set_deviation_scalar(vnames, cp['poly_constraints'], pl, pu)
        return vl, sp

    if out > 0:
        vl, sp = _set000_obj()
        sp_str = ', \\\\\n& \\quad '.join(sp)
        L.append(f"\\mathcal{{U}}_{{c}} = \\{{ {vl} : \\quad & {sp_str} \\}} \\\\")

    for i in range(nc):
        ct, cp, ui = cts[i], cps[i], cui[i]
        if ct == 0: continue
        ci = i + 1
        vl, sp = _set000_con(ci, ct, cp, ui)
        sp_str = ', \\\\\n& \\quad '.join(sp)
        L.append(f"\\mathcal{{U}}_{{a{ci}}} = \\{{ {vl} : \\quad & {sp_str} \\}} \\\\")

    L.append("\\end{align*}")
    return "\n".join(L)


def _template_001(params):
    nv, nc, oc, od, out, oup, cts, cps, oui, cui, nr, pr, xl, xu, pl, pu = _extract_params(params)
    L = ["\\begin{align*}"]
    opt = "\\max" if od else "\\min"

    _base_of = {1: "box", 2: "budg", 3: "poly"}
    _con_type_ct = {}
    for ct in cts:
        if ct > 0:
            _con_type_ct[ct] = _con_type_ct.get(ct, 0) + 1

    def _sn001_con(base, ct, ci):
        if _con_type_ct[ct] > 1:
            return f"\\mathcal{{U}}_{{{base}\\_con{ci}}}"
        return f"\\mathcal{{U}}_{{{base}\\_con}}"


    if out == 0:
        L.append(f"{opt} \\quad & {_nominal_expr(nv, oc)} \\quad & \\text{{Objective: Deterministic}} \\\\")
    else:
        terms = []
        first = True
        for j in range(nv):
            c = oc[j]
            if j in oui:
                sym = f"\\delta_{{{out + 1}{j + 1}}}"
                if first:
                    terms.append(f"({_fmt(c)} + {sym})x_{{{j + 1}}}")
                else:
                    terms.append(f"+ ({_fmt(c)} + {sym})x_{{{j + 1}}}")
            else:
                if abs(c) < 1e-10: continue
                if first and c >= 0:
                    terms.append(f"{_fmt(abs(c))}x_{{{j + 1}}}")
                else:
                    terms.append(f"{'+ ' if c >= 0 else '- '}{_fmt(abs(c))}x_{{{j + 1}}}")
            first = False

        expr = " ".join(terms).replace("+ -", "- ")
        vl = "(" + ", ".join([f"\\delta_{{{out + 1}{j + 1}}}" for j in oui]) + ")"
        sn = f"\\mathcal{{U}}_{{{_base_of[out]}\\_obj}}"
        lbl = {1: "Box Uncertainty", 2: "Budgeted Uncertainty", 3: "Polyhedral Uncertainty"}[out]
        L.append(f"{opt} \\quad & {expr} \\quad \\forall {vl} \\in {sn} \\quad & \\text{{Objective: {lbl}}} \\\\")


    for i in range(nc):
        ct, cp, ui = cts[i], cps[i], cui[i]
        ci = i + 1
        iq = _ineq(cp['direction'])

        prefix = "\\text{s.t.} \\quad & " if i == 0 else "& "

        if ct == 0:
            L.append(
                f"{prefix}{_nominal_expr(nv, cp['nominal_coeffs'])} {iq} {_fmt(cp['b'])} \\quad & \\text{{Constraint {ci}: Deterministic}} \\\\")
        else:
            terms = []
            first = True
            for j in range(nv):
                c = cp['nominal_coeffs'][j]
                if j in ui:
                    sym = f"\\epsilon_{{{ct + 1}{j + 1}}}"
                    if first:
                        terms.append(f"({_fmt(c)} + {sym})x_{{{j + 1}}}")
                    else:
                        terms.append(f"+ ({_fmt(c)} + {sym})x_{{{j + 1}}}")
                else:
                    if abs(c) < 1e-10: continue
                    if first and c >= 0:
                        terms.append(f"{_fmt(abs(c))}x_{{{j + 1}}}")
                    else:
                        terms.append(f"{'+ ' if c >= 0 else '- '}{_fmt(abs(c))}x_{{{j + 1}}}")
                first = False

            expr = " ".join(terms).replace("+ -", "- ")
            vl = "(" + ", ".join([f"\\epsilon_{{{ct + 1}{j + 1}}}" for j in ui]) + ")"
            sn = _sn001_con(_base_of[ct], ct, ci)
            lbl = {1: "Box Uncertainty", 2: "Budgeted Uncertainty", 3: "Polyhedral Uncertainty"}[ct]
            L.append(
                f"{prefix}{expr} {iq} {_fmt(cp['b'])} \\quad \\forall {vl} \\in {sn} \\quad & \\text{{Constraint {ci}: {lbl}}} \\\\")


    bnd = ", ".join([_x_bnd_str(j, xl, xu) for j in range(nv)])
    L.append(f"& {bnd} \\\\")


    def _sp001_build(syms, ct_type, cp_params):
        if ct_type == 3:
            sp = _poly_set_deviation_scalar(syms, cp_params['poly_constraints'], pl, pu)
        elif ct_type == 1:
            sp = [f"-{_fmt(cp_params['deviation'][k])} \\le {syms[k]} \\le {_fmt(cp_params['deviation'][k])}" for k in
                  range(len(syms))]
        else:
            sp = [f"|{syms[k]}| \\le {_fmt(cp_params['deviation'][k])}" for k in range(len(syms))]
            sp.append(" + ".join([f"\\frac{{|{syms[k]}|}}{{{_fmt(cp_params['deviation'][k])}}}" for k in
                                  range(len(syms))]) + f" \\le {_fmt(cp_params['gamma'])}")
        return sp

    if out > 0:
        syms = [f"\\delta_{{{out + 1}{j + 1}}}" for j in oui]
        vl = "(" + ", ".join(syms) + ")"
        sn = f"\\mathcal{{U}}_{{{_base_of[out]}\\_obj}}"
        sp = _sp001_build(syms, out, oup)

        L.append(f"& {sn} = \\{{ {vl} : {', '.join(sp)} \\}} \\\\")

    for i in range(nc):
        ct, cp, ui = cts[i], cps[i], cui[i]
        if ct == 0: continue
        ci = i + 1
        syms = [f"\\epsilon_{{{ct + 1}{j + 1}}}" for j in ui]
        vl = "(" + ", ".join(syms) + ")"
        sn = _sn001_con(_base_of[ct], ct, ci)
        sp = _sp001_build(syms, ct, cp)
        L.append(f"& {sn} = \\{{ {vl} : {', '.join(sp)} \\}} \\\\")

    L.append("\\end{align*}")
    return "\n".join(L)


def _template_010(params):
    nv, nc, oc, od, out, oup, cts, cps, oui, cui, nr, pr, xl, xu, pl, pu = _extract_params(params)
    L = ["\\begin{align*}"]
    opt = "\\max_{\\mathbf{x}}" if od else "\\min_{\\mathbf{x}}"

    def _vec(v):
        return "[" + ", ".join([_fmt(x) for x in v]) + "]^\\top"

    def _mat(P):
        rows = []
        for row in P:
            rows.append(" & ".join([_fmt(x) for x in row]))
        return "\\begin{bmatrix} " + " \\\\ ".join(rows) + " \\end{bmatrix}"

    def _poly_mat_vec(ui, poly_constraints, pl_val, pu_val, nv):
        P_rows = []
        q_vals = []
        E_rows = []
        e_vals = []
        for A, b, direction in poly_constraints:
            row = [0] * nv
            for k, j in enumerate(ui):
                row[j] = A[k]
            if direction == 1:
                P_rows.append(row)
                q_vals.append(b)
            elif direction == -1:
                P_rows.append([-x for x in row])
                q_vals.append(-b)
            else:
                E_rows.append(row)
                e_vals.append(b)
        for j in ui:
            row = [0] * nv
            row[j] = 1
            P_rows.append(row)
            q_vals.append(pu_val)
            row_neg = [0] * nv
            row_neg[j] = -1
            P_rows.append(row_neg)
            q_vals.append(-pl_val)

        det_indices = [j for j in range(nv) if j not in ui]
        if det_indices:
            for j in det_indices:
                row_eq = [0] * nv
                row_eq[j] = 1
                E_rows.append(row_eq)
                e_vals.append(0)

        if len(E_rows) > 0:
            return _mat(P_rows), _vec(q_vals), _mat(E_rows), _vec(e_vals)
        return _mat(P_rows), _vec(q_vals), None, None

    defs = []


    if out == 0:
        L.append(f"{opt} \\quad & \\mathbf{{c}}^\\top \\mathbf{{x}} \\\\")
        defs.append(f"\\mathbf{{c}} = {_vec(oc)}")
    elif out == 3:
        cb = "\\boldsymbol{\\delta}_c"
        L.append(
            f"{opt} \\quad & (\\mathbf{{c}}^{{nom}} + {cb})^\\top \\mathbf{{x}}, \\quad \\forall {cb} \\in \\mathcal{{U}}_c \\\\")
        defs.append(f"\\mathbf{{c}}^{{nom}} = {_vec(oc)}")
    else:
        L.append(
            f"{opt} \\quad & \\mathbf{{c}}^\\top \\mathbf{{x}}, \\quad \\forall \\mathbf{{c}} \\in \\mathcal{{U}}_c \\\\")
        defs.append(f"\\mathbf{{c}}^{{nom}} = {_vec(oc)}")


    for i in range(nc):
        ct, cp = cts[i], cps[i]
        ci = i + 1
        iq = _ineq(cp['direction'])
        if ct == 0:
            L.append(f"& \\mathbf{{a}}_{{{ci}}}^\\top \\mathbf{{x}} {iq} b_{{{ci}}} \\\\")
            defs.append(f"\\mathbf{{a}}_{{{ci}}} = {_vec(cp['nominal_coeffs'])}")
            defs.append(f"b_{{{ci}}} = {_fmt(cp['b'])}")
        elif ct == 3:
            eb = f"\\boldsymbol{{\\delta}}_{{{ci}}}"
            L.append(
                f"& (\\mathbf{{a}}_{{{ci}}}^{{nom}} + {eb})^\\top \\mathbf{{x}} {iq} b_{{{ci}}}, \\quad \\forall {eb} \\in \\mathcal{{U}}_{{{ci}}} \\\\")
            defs.append(f"\\mathbf{{a}}_{{{ci}}}^{{nom}} = {_vec(cp['nominal_coeffs'])}")
            defs.append(f"b_{{{ci}}} = {_fmt(cp['b'])}")
        else:
            L.append(
                f"& \\mathbf{{a}}_{{{ci}}}^\\top \\mathbf{{x}} {iq} b_{{{ci}}}, \\quad \\forall \\mathbf{{a}}_{{{ci}}} \\in \\mathcal{{U}}_{{{ci}}} \\\\")
            defs.append(f"\\mathbf{{a}}_{{{ci}}}^{{nom}} = {_vec(cp['nominal_coeffs'])}")
            defs.append(f"b_{{{ci}}} = {_fmt(cp['b'])}")

    bnd = _x_bnd_compact(xl, xu, nv)
    defs.append(bnd)


    if out > 0:
        cb = "\\boldsymbol{\\delta}_c"
        if out == 3:
            P_mat, q_vec, E_mat, e_vec = _poly_mat_vec(oui, oup['poly_constraints'], pl, pu, nv)
            if E_mat is not None:
                L.append(
                    f"& \\mathcal{{U}}_c = \\{{ {cb} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{P}}_c {cb} \\le \\mathbf{{q}}_c, \\quad \\mathbf{{E}}_c {cb} = \\mathbf{{e}}_c \\}} \\\\")
                defs.append(f"\\mathbf{{P}}_c = {P_mat}")
                defs.append(f"\\mathbf{{q}}_c = {q_vec}")
                defs.append(f"\\mathbf{{E}}_c = {E_mat}")
                defs.append(f"\\mathbf{{e}}_c = {e_vec}")
            else:
                L.append(
                    f"& \\mathcal{{U}}_c = \\{{ {cb} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{P}}_c {cb} \\le \\mathbf{{q}}_c \\}} \\\\")
                defs.append(f"\\mathbf{{P}}_c = {P_mat}")
                defs.append(f"\\mathbf{{q}}_c = {q_vec}")
        elif out == 1:
            L.append(
                f"& \\mathcal{{U}}_c = \\{{ \\mathbf{{c}} \\in \\mathbb{{R}}^{{{nv}}} : |\\mathbf{{c}} - \\mathbf{{c}}^{{nom}}| \\le \\hat{{\\mathbf{{c}}}} \\}} \\\\")
            dev_vec = [0.0] * nv
            for k, j in enumerate(oui): dev_vec[j] = oup['deviation'][k]
            defs.append(f"\\hat{{\\mathbf{{c}}}} = {_vec(dev_vec)}")
        elif out == 2:
            ui_set = "\\{" + ", ".join([str(x + 1) for x in oui]) + "\\}"
            L.append(
                f"& \\mathcal{{U}}_c = \\{{ \\mathbf{{c}} \\in \\mathbb{{R}}^{{{nv}}} : |\\mathbf{{c}} - \\mathbf{{c}}^{{nom}}| \\le \\hat{{\\mathbf{{c}}}}, \\quad \\sum_{{j \\in {ui_set}}} \\frac{{| c_{{j}} - c_j^{{nom}} |}}{{\\hat{{c}}_j}} \\le \\Gamma_c \\}} \\\\")
            dev_vec = [0.0] * nv
            for k, j in enumerate(oui): dev_vec[j] = oup['deviation'][k]
            defs.append(f"\\hat{{\\mathbf{{c}}}} = {_vec(dev_vec)}")
            defs.append(f"\\Gamma_c = {_fmt(oup['gamma'])}")

    for i in range(nc):
        ct, cp, ui = cts[i], cps[i], cui[i]
        if ct == 0: continue
        ci = i + 1
        sn = f"\\mathcal{{U}}_{{{ci}}}"
        if ct == 3:
            eb = f"\\boldsymbol{{\\delta}}_{{{ci}}}"
            P_mat, q_vec, E_mat, e_vec = _poly_mat_vec(ui, cp['poly_constraints'], pl, pu, nv)
            if E_mat is not None:
                L.append(
                    f"& {sn} = \\{{ {eb} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{P}}_{{{ci}}} {eb} \\le \\mathbf{{q}}_{{{ci}}}, \\quad \\mathbf{{E}}_{{{ci}}} {eb} = \\mathbf{{e}}_{{{ci}}} \\}} \\\\")
                defs.append(f"\\mathbf{{P}}_{{{ci}}} = {P_mat}")
                defs.append(f"\\mathbf{{q}}_{{{ci}}} = {q_vec}")
                defs.append(f"\\mathbf{{E}}_{{{ci}}} = {E_mat}")
                defs.append(f"\\mathbf{{e}}_{{{ci}}} = {e_vec}")
            else:
                L.append(
                    f"& {sn} = \\{{ {eb} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{P}}_{{{ci}}} {eb} \\le \\mathbf{{q}}_{{{ci}}} \\}} \\\\")
                defs.append(f"\\mathbf{{P}}_{{{ci}}} = {P_mat}")
                defs.append(f"\\mathbf{{q}}_{{{ci}}} = {q_vec}")
        elif ct == 1:
            L.append(
                f"& {sn} = \\{{ \\mathbf{{a}}_{{{ci}}} \\in \\mathbb{{R}}^{{{nv}}} : |\\mathbf{{a}}_{{{ci}}} - \\mathbf{{a}}_{{{ci}}}^{{nom}}| \\le \\hat{{\\mathbf{{a}}}}_{{{ci}}} \\}} \\\\")
            dev_vec = [0.0] * nv
            for k, j in enumerate(ui): dev_vec[j] = cp['deviation'][k]
            defs.append(f"\\hat{{\\mathbf{{a}}}}_{{{ci}}} = {_vec(dev_vec)}")
        elif ct == 2:
            ui_set = "\\{" + ", ".join([str(x + 1) for x in ui]) + "\\}"
            L.append(
                f"& {sn} = \\{{ \\mathbf{{a}}_{{{ci}}} \\in \\mathbb{{R}}^{{{nv}}} : |\\mathbf{{a}}_{{{ci}}} - \\mathbf{{a}}_{{{ci}}}^{{nom}}| \\le \\hat{{\\mathbf{{a}}}}_{{{ci}}}, \\quad \\sum_{{j \\in {ui_set}}} \\frac{{| a_{{{ci},j}} - a_{{{ci},j}}^{{nom}} |}}{{\\hat{{a}}_{{{ci},j}}}} \\le \\Gamma_{{{ci}}} \\}} \\\\")
            dev_vec = [0.0] * nv
            for k, j in enumerate(ui): dev_vec[j] = cp['deviation'][k]
            defs.append(f"\\hat{{\\mathbf{{a}}}}_{{{ci}}} = {_vec(dev_vec)}")
            defs.append(f"\\Gamma_{{{ci}}} = {_fmt(cp['gamma'])}")

    x_vec_str = "[" + ", ".join([f"x_{{{j + 1}}}" for j in range(nv)]) + "]^\\top"
    L.append(f"& \\mathbf{{x}} = {x_vec_str} \\\\")

    for d in defs:
        L.append(f"& {d} \\\\")
    L.append("\\end{align*}")
    return "\n".join(L)


def _template_011(params):
    nv, nc, oc, od, out, oup, cts, cps, oui, cui, nr, pr, xl, xu, pl, pu = _extract_params(params)
    L = ["\\begin{align*}"]
    opt = "\\text{Maximize}" if od else "\\text{Minimize}"

    def _vec(v):
        return "[" + ", ".join([_fmt(x) for x in v]) + "]^\\top"

    def _mat(P):
        rows = []
        for row in P:
            rows.append(" & ".join([_fmt(x) for x in row]))
        return "\\begin{bmatrix} " + " \\\\ ".join(rows) + " \\end{bmatrix}"

    def _poly_mat_vec(ui, poly_constraints, pl_val, pu_val, nv):
        P_rows = []
        q_vals = []
        E_rows = []
        e_vals = []
        for A, b, direction in poly_constraints:
            row = [0] * nv
            for k, j in enumerate(ui):
                row[j] = A[k]
            if direction == 1:
                P_rows.append(row)
                q_vals.append(b)
            elif direction == -1:
                P_rows.append([-x for x in row])
                q_vals.append(-b)
            else:
                E_rows.append(row)
                e_vals.append(b)
        for j in ui:
            row = [0] * nv
            row[j] = 1
            P_rows.append(row)
            q_vals.append(pu_val)
            row_neg = [0] * nv
            row_neg[j] = -1
            P_rows.append(row_neg)
            q_vals.append(-pl_val)

        det_indices = [j for j in range(nv) if j not in ui]
        if det_indices:
            for j in det_indices:
                row_eq = [0] * nv
                row_eq[j] = 1
                E_rows.append(row_eq)
                e_vals.append(0)

        if len(E_rows) > 0:
            return _mat(P_rows), _vec(q_vals), _mat(E_rows), _vec(e_vals)
        return _mat(P_rows), _vec(q_vals), None, None

    def _diag(devs, ui, nv):
        d = [0.0] * nv
        for k, j in enumerate(ui): d[j] = devs[k]
        return "\\text{diag}(" + ", ".join([_fmt(x) for x in d]) + ")"

    defs = []


    _t011_type_ct = {}
    for ct in ([out] if out > 0 else []) + [ct for ct in cts if ct > 0]:
        _t011_type_ct[ct] = _t011_type_ct.get(ct, 0) + 1

    def _sn011_obj(stype):
        return f"\\mathcal{{U}}_{{c,{stype}}}"

    def _sn011_con(stype, ci=None):
        key = {'Box': 1, 'Bud': 2, 'P': 3}.get(stype, 0)
        multi = _t011_type_ct.get(key, 1) > 1
        if ci is not None and multi:
            return f"\\mathcal{{U}}_{{a{ci},{stype}}}"
        return f"\\mathcal{{U}}_{{a,{stype}}}"


    if out == 0:
        L.append(f"{opt} \\quad & \\mathbf{{c}}^\\top \\mathbf{{x}} && && \\text{{(Deterministic Objective)}} \\\\")
        defs.append(f"\\mathbf{{c}} = {_vec(oc)}")
    elif out == 3:
        cb = "\\boldsymbol{\\delta}_c"
        L.append(
            f"{opt} \\quad & (\\mathbf{{c}}^{{nom}} + {cb})^\\top \\mathbf{{x}} && \\forall {cb} \\in {_sn011_obj('P')} && \\text{{(Polyhedral Uncertainty Objective)}} \\\\")
        defs.append(f"\\mathbf{{c}}^{{nom}} = {_vec(oc)}")
    else:
        sn_obj = _sn011_obj('Box') if out == 1 else _sn011_obj('Bud')
        lbl = "Box Uncertainty Objective" if out == 1 else "Budgeted Uncertainty Objective"
        L.append(
            f"{opt} \\quad & \\mathbf{{c}}^\\top \\mathbf{{x}} && \\forall \\mathbf{{c}} \\in {sn_obj} && \\text{{({lbl})}} \\\\")
        defs.append(f"\\mathbf{{c}}^{{nom}} = {_vec(oc)}")


    for i in range(nc):
        ct, cp = cts[i], cps[i]
        ci = i + 1
        iq = _ineq(cp['direction'])
        prefix = "\\text{subject to} \\quad &" if i == 0 else "&"
        if ct == 0:
            L.append(
                f"{prefix} \\mathbf{{a}}_{{{ci}}}^\\top \\mathbf{{x}} {iq} b_{{{ci}}} && && \\text{{(Deterministic Constraint)}} \\\\")
            defs.append(f"\\mathbf{{a}}_{{{ci}}} = {_vec(cp['nominal_coeffs'])}")
            defs.append(f"b_{{{ci}}} = {_fmt(cp['b'])}")
        elif ct == 3:
            eb = f"\\boldsymbol{{\\delta}}_{{a{ci}}}"
            sn = _sn011_con('P', ci)
            L.append(
                f"{prefix} (\\mathbf{{a}}_{{{ci}}}^{{nom}} + {eb})^\\top \\mathbf{{x}} {iq} b_{{{ci}}} && \\forall {eb} \\in {sn} && \\text{{(Polyhedral Uncertainty Constraint)}} \\\\")
            defs.append(f"\\mathbf{{a}}_{{{ci}}}^{{nom}} = {_vec(cp['nominal_coeffs'])}")
            defs.append(f"b_{{{ci}}} = {_fmt(cp['b'])}")
        else:
            stype = 'Box' if ct == 1 else 'Bud'
            lbl = "Box Uncertainty Constraint" if ct == 1 else "Budgeted Uncertainty Constraint"
            sn = _sn011_con(stype, ci)
            L.append(
                f"{prefix} \\mathbf{{a}}_{{{ci}}}^\\top \\mathbf{{x}} {iq} b_{{{ci}}} && \\forall \\mathbf{{a}}_{{{ci}}} \\in {sn} && \\text{{({lbl})}} \\\\")
            defs.append(f"\\mathbf{{a}}_{{{ci}}}^{{nom}} = {_vec(cp['nominal_coeffs'])}")
            defs.append(f"b_{{{ci}}} = {_fmt(cp['b'])}")


    has_unc = out > 0 or any(ct > 0 for ct in cts)
    if has_unc:
        L.append("\\text{where:} \\quad")

    if out > 0:
        if out == 3:
            sn = _sn011_obj('P')
            cb = "\\boldsymbol{\\delta}_c"
            P_mat, q_vec, E_mat, e_vec = _poly_mat_vec(oui, oup['poly_constraints'], pl, pu, nv)
            if E_mat is not None:
                L.append(
                    f"& {sn} = \\{{ {cb} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{M}}_c {cb} \\le \\mathbf{{q}}_c, \\quad \\mathbf{{E}}_c {cb} = \\mathbf{{e}}_c \\}} \\\\")
                defs.append(f"\\mathbf{{M}}_c = {P_mat}")
                defs.append(f"\\mathbf{{q}}_c = {q_vec}")
                defs.append(f"\\mathbf{{E}}_c = {E_mat}")
                defs.append(f"\\mathbf{{e}}_c = {e_vec}")
            else:
                L.append(
                    f"& {sn} = \\{{ {cb} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{M}}_c {cb} \\le \\mathbf{{q}}_c \\}} \\\\")
                defs.append(f"\\mathbf{{M}}_c = {P_mat}")
                defs.append(f"\\mathbf{{q}}_c = {q_vec}")
        elif out == 1:
            sn = _sn011_obj('Box')
            L.append(
                f"& {sn} = \\{{ \\mathbf{{c}} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{c}} = \\mathbf{{c}}^{{nom}} + \\mathbf{{D}}_c \\boldsymbol{{\\xi}}_c, \\|\\boldsymbol{{\\xi}}_c\\|_{{\\infty}} \\le 1 \\}} \\\\")
            defs.append(f"\\mathbf{{D}}_c = {_diag(oup['deviation'], oui, nv)}")
        elif out == 2:
            sn = _sn011_obj('Bud')
            L.append(
                f"& {sn} = \\{{ \\mathbf{{c}} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{c}} = \\mathbf{{c}}^{{nom}} + \\mathbf{{D}}_c \\boldsymbol{{\\xi}}_c, \\|\\boldsymbol{{\\xi}}_c\\|_{{\\infty}} \\le 1, \\|\\boldsymbol{{\\xi}}_c\\|_1 \\le \\Gamma_c \\}} \\\\")
            defs.append(f"\\mathbf{{D}}_c = {_diag(oup['deviation'], oui, nv)}")
            defs.append(f"\\Gamma_c = {_fmt(oup['gamma'])}")

    for i in range(nc):
        ct, cp, ui = cts[i], cps[i], cui[i]
        if ct == 0: continue
        ci = i + 1
        if ct == 3:
            sn = _sn011_con('P', ci)
            eb = f"\\boldsymbol{{\\delta}}_{{a{ci}}}"
            P_mat, q_vec, E_mat, e_vec = _poly_mat_vec(ui, cp['poly_constraints'], pl, pu, nv)
            if E_mat is not None:
                L.append(
                    f"& {sn} = \\{{ {eb} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{M}}_{{a{ci}}} {eb} \\le \\mathbf{{q}}_{{a{ci}}}, \\quad \\mathbf{{E}}_{{a{ci}}} {eb} = \\mathbf{{e}}_{{a{ci}}} \\}} \\\\")
                defs.append(f"\\mathbf{{M}}_{{a{ci}}} = {P_mat}")
                defs.append(f"\\mathbf{{q}}_{{a{ci}}} = {q_vec}")
                defs.append(f"\\mathbf{{E}}_{{a{ci}}} = {E_mat}")
                defs.append(f"\\mathbf{{e}}_{{a{ci}}} = {e_vec}")
            else:
                L.append(
                    f"& {sn} = \\{{ {eb} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{M}}_{{a{ci}}} {eb} \\le \\mathbf{{q}}_{{a{ci}}} \\}} \\\\")
                defs.append(f"\\mathbf{{M}}_{{a{ci}}} = {P_mat}")
                defs.append(f"\\mathbf{{q}}_{{a{ci}}} = {q_vec}")
        elif ct == 1:
            sn = _sn011_con('Box', ci)
            L.append(
                f"& {sn} = \\{{ \\mathbf{{a}}_{{{ci}}} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{a}}_{{{ci}}} = \\mathbf{{a}}_{{{ci}}}^{{nom}} + \\mathbf{{D}}_{{a{ci}}} \\boldsymbol{{\\xi}}_{{a{ci}}}, \\|\\boldsymbol{{\\xi}}_{{a{ci}}}\\|_{{\\infty}} \\le 1 \\}} \\\\")
            defs.append(f"\\mathbf{{D}}_{{a{ci}}} = {_diag(cp['deviation'], ui, nv)}")
        elif ct == 2:
            sn = _sn011_con('Bud', ci)
            L.append(
                f"& {sn} = \\{{ \\mathbf{{a}}_{{{ci}}} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{a}}_{{{ci}}} = \\mathbf{{a}}_{{{ci}}}^{{nom}} + \\mathbf{{D}}_{{a{ci}}} \\boldsymbol{{\\xi}}_{{a{ci}}}, \\|\\boldsymbol{{\\xi}}_{{a{ci}}}\\|_{{\\infty}} \\le 1, \\|\\boldsymbol{{\\xi}}_{{a{ci}}}\\|_1 \\le \\Gamma_{{a{ci}}} \\}} \\\\")
            defs.append(f"\\mathbf{{D}}_{{a{ci}}} = {_diag(cp['deviation'], ui, nv)}")
            defs.append(f"\\Gamma_{{a{ci}}} = {_fmt(cp['gamma'])}")

    x_vec_str = "[" + ", ".join([f"x_{{{j + 1}}}" for j in range(nv)]) + "]^\\top"
    L.append(f"& \\mathbf{{x}} = {x_vec_str} \\\\")

    for d in defs:
        L.append(f"& {d} \\\\")

    prefix = "\\text{subject to} \\quad &" if nc == 0 else "&"
    bnd = _x_bnd_compact(xl, xu, nv)
    L.append(f"{prefix} {bnd} && && \\text{{(Variable Bounds)}} \\\\")
    L.append("\\end{align*}")
    return "\n".join(L)


def _template_100(params):
    nv, nc, oc, od, out, oup, cts, cps, oui, cui, nr, pr, xl, xu, pl, pu = _extract_params(params)
    L = ["\\begin{align*}"]
    xvars = ", ".join([f"x_{{{j + 1}}}" for j in range(nv)])
    opt = "\\max_{{{}}}".format(xvars) if od else "\\min_{{{}}}".format(xvars)


    if out == 0:
        L.append(f"{opt} \\quad & {_nominal_expr(nv, oc)} \\\\")
    else:
        op_inner = "\\min" if od else "\\max"

        if out == 3:
            terms = []
            for j in range(nv):
                c = oc[j]
                if j in oui:
                    if abs(c) > 1e-10: terms.append(f"{_fmt(c)}x_{{{j + 1}}}")
                    terms.append(f"\\delta^c_{{{j + 1}}}x_{{{j + 1}}}")
                else:
                    if abs(c) > 1e-10: terms.append(f"{_fmt(c)}x_{{{j + 1}}}")
            expr = " + ".join(terms).replace("+ -", "- ")
            vl = ", ".join([f"\\delta^c_{{{j + 1}}}" for j in oui])
            L.append(f"{opt} \\quad & {op_inner}_{{{vl} \\in \\mathcal{{U}}^c}} {expr} \\\\")
        else:
            terms = []
            for j in range(nv):
                c = oc[j]
                if j in oui:
                    k = oui.index(j)
                    hat = oup['deviation'][k]
                    if abs(c) > 1e-10: terms.append(f"{_fmt(c)}x_{{{j + 1}}}")
                    terms.append(f"{_fmt(hat)}\\xi^c_{{{j + 1}}}x_{{{j + 1}}}")
                else:
                    if abs(c) > 1e-10: terms.append(f"{_fmt(c)}x_{{{j + 1}}}")
            expr = " + ".join(terms).replace("+ -", "- ")
            vl = ", ".join([f"\\xi^c_{{{j + 1}}}" for j in oui])
            L.append(f"{opt} \\quad & {op_inner}_{{{vl} \\in \\mathcal{{U}}^c}} {expr} \\\\")


    for i in range(nc):
        ct, cp, ui = cts[i], cps[i], cui[i]
        ci = i + 1
        iq = _ineq(cp['direction'])
        if ct == 0:
            L.append(f"& {_nominal_expr(nv, cp['nominal_coeffs'])} {iq} {_fmt(cp['b'])} \\\\")
        elif ct == 3:
            op_inner = "\\max" if cp['direction'] == 1 else ("\\min" if cp['direction'] == -1 else "\\text{eval}")
            terms = []
            for j in range(nv):
                c = cp['nominal_coeffs'][j]
                if j in ui:
                    if abs(c) > 1e-10: terms.append(f"{_fmt(c)}x_{{{j + 1}}}")
                    terms.append(f"\\delta^a_{{{ci},{j + 1}}}x_{{{j + 1}}}")
                else:
                    if abs(c) > 1e-10: terms.append(f"{_fmt(c)}x_{{{j + 1}}}")
            expr = " + ".join(terms).replace("+ -", "- ")
            vl = ", ".join([f"\\delta^a_{{{ci},{j + 1}}}" for j in ui])
            L.append(f"& {op_inner}_{{{vl} \\in \\mathcal{{U}}^a_{{{ci}}}}} {expr} {iq} {_fmt(cp['b'])} \\\\")
        else:
            op_inner = "\\max" if cp['direction'] == 1 else ("\\min" if cp['direction'] == -1 else "\\text{eval}")
            terms = []
            for j in range(nv):
                c = cp['nominal_coeffs'][j]
                if j in ui:
                    k = ui.index(j)
                    hat = cp['deviation'][k]
                    if abs(c) > 1e-10: terms.append(f"{_fmt(c)}x_{{{j + 1}}}")
                    terms.append(f"{_fmt(hat)}\\xi^a_{{{ci},{j + 1}}}x_{{{j + 1}}}")
                else:
                    if abs(c) > 1e-10: terms.append(f"{_fmt(c)}x_{{{j + 1}}}")
            expr = " + ".join(terms).replace("+ -", "- ")
            vl = ", ".join([f"\\xi^a_{{{ci},{j + 1}}}" for j in ui])
            L.append(f"& {op_inner}_{{{vl} \\in \\mathcal{{U}}^a_{{{ci}}}}} {expr} {iq} {_fmt(cp['b'])} \\\\")


    if out > 0:
        if out == 3:
            vl = ", ".join([f"\\delta^c_{{{j + 1}}}" for j in oui])
            sp = _poly_set_deviation_scalar([f"\\delta^c_{{{j + 1}}}" for j in oui], oup['poly_constraints'], pl, pu)
            L.append(f"& \\mathcal{{U}}^c = \\{{ {vl} : {', '.join(sp)} \\}} \\\\")
        elif out == 1:
            vl = ", ".join([f"\\xi^c_{{{j + 1}}}" for j in oui])
            sp = [f"-1 \\le \\xi^c_{{{j + 1}}} \\le 1" for j in oui]
            L.append(f"& \\mathcal{{U}}^c = \\{{ {vl} : {', '.join(sp)} \\}} \\\\")
        else:
            vl = ", ".join([f"\\xi^c_{{{j + 1}}}" for j in oui])
            sp_budget = " + ".join([f"|\\xi^c_{{{j + 1}}}|" for j in oui]) + f" \\le {_fmt(oup['gamma'])}"
            sp = [f"-1 \\le \\xi^c_{{{j + 1}}} \\le 1" for j in oui]
            L.append(f"& \\mathcal{{U}}^c = \\{{ {vl} : {sp_budget}, {', '.join(sp)} \\}} \\\\")

    for i in range(nc):
        ct, cp, ui = cts[i], cps[i], cui[i]
        if ct == 0: continue
        ci = i + 1
        if ct == 3:
            vl = ", ".join([f"\\delta^a_{{{ci},{j + 1}}}" for j in ui])
            sp = _poly_set_deviation_scalar([f"\\delta^a_{{{ci},{j + 1}}}" for j in ui], cp['poly_constraints'], pl, pu)
            L.append(f"& \\mathcal{{U}}^a_{{{ci}}} = \\{{ {vl} : {', '.join(sp)} \\}} \\\\")
        elif ct == 1:
            vl = ", ".join([f"\\xi^a_{{{ci},{j + 1}}}" for j in ui])
            sp = [f"-1 \\le \\xi^a_{{{ci},{j + 1}}} \\le 1" for j in ui]
            L.append(f"& \\mathcal{{U}}^a_{{{ci}}} = \\{{ {vl} : {', '.join(sp)} \\}} \\\\")
        else:
            vl = ", ".join([f"\\xi^a_{{{ci},{j + 1}}}" for j in ui])
            sp_budget = " + ".join([f"|\\xi^a_{{{ci},{j + 1}}}|" for j in ui]) + f" \\le {_fmt(cp['gamma'])}"
            sp = [f"-1 \\le \\xi^a_{{{ci},{j + 1}}} \\le 1" for j in ui]
            L.append(f"& \\mathcal{{U}}^a_{{{ci}}} = \\{{ {vl} : {sp_budget}, {', '.join(sp)} \\}} \\\\")

    bnd = ", ".join([_x_bnd_str(j, xl, xu) for j in range(nv)])
    L.append(f"& {bnd} \\\\")
    L.append("\\end{align*}")
    return "\n".join(L)


def _template_101(params):
    nv, nc, oc, od, out, oup, cts, cps, oui, cui, nr, pr, xl, xu, pl, pu = _extract_params(params)
    L = ["\\begin{align*}"]
    xvars = ", ".join([f"x_{{{j + 1}}}" for j in range(nv)])
    opt = "\\text{maximize }" if od else "\\text{minimize }"

    def _expr_terms(coeffs, ui, var_symbol, is_box=False):
        terms = []
        for j in range(nv):
            c = coeffs[j]
            if j in ui:
                if is_box:
                    k = ui.index(j)
                    hat = oup['deviation'][k] if coeffs is oc else cps[ui[0] - 1 if len(ui) > 0 else 0]['deviation'][
                        k] if 'deviation' in cp else 0

            pass
        return terms

    def _format_expr(coeffs, ui, dev_array, symbol):
        terms = []
        for j in range(nv):
            c = coeffs[j]
            if j in ui:
                k = ui.index(j)
                if dev_array is not None:
                    hat = dev_array[k]
                    terms.append(f"({_fmt(c)} + {_fmt(hat)}{symbol}_{{{j + 1}}})x_{{{j + 1}}}")
                else:
                    terms.append(f"({_fmt(c)} + {symbol}_{{{j + 1}}})x_{{{j + 1}}}")
            else:
                if abs(c) > 1e-10: terms.append(f"{_fmt(c)}x_{{{j + 1}}}")
        return " + ".join(terms).replace("+ -", "- ")

    def _format_con_expr(ci, coeffs, ui, dev_array, symbol):
        terms = []
        for j in range(nv):
            c = coeffs[j]
            if j in ui:
                k = ui.index(j)
                if dev_array is not None:
                    hat = dev_array[k]
                    terms.append(f"({_fmt(c)} + {_fmt(hat)}{symbol}_{{{j + 1}}})x_{{{j + 1}}}")
                else:
                    terms.append(f"({_fmt(c)} + {symbol}_{{{ci},{j + 1}}})x_{{{j + 1}}}")
            else:
                if abs(c) > 1e-10: terms.append(f"{_fmt(c)}x_{{{j + 1}}}")
        return " + ".join(terms).replace("+ -", "- ")


    if out == 0:
        L.append(f"& {opt} {_nominal_expr(nv, oc)} && \\text{{[Deterministic Objective]}} \\\\")
    else:
        op_inner = "\\min" if od else "\\max"
        if out == 3:
            vl = ", ".join([f"\\delta^c_{{{j + 1}}}" for j in oui])
            expr = _format_expr(oc, oui, None, "\\delta^c")
            L.append(
                f"& {opt} {op_inner}_{{{vl} \\in \\mathcal{{U}}^c_{{poly}}}} \\{{ {expr} \\}} && \\text{{[Polyhedral Uncertainty Objective]}} \\\\")
        elif out == 1:
            bounds = ", ".join([f"-1 \\le \\xi^c_{{{j + 1}}} \\le 1" for j in oui])
            expr = _format_expr(oc, oui, oup['deviation'], "\\xi^c")
            L.append(f"& {opt} {op_inner}_{{{bounds}}} \\{{ {expr} \\}} && \\text{{[Box Uncertainty Objective]}} \\\\")
        else:
            vl = ", ".join([f"\\eta^c_{{{j + 1}}}" for j in oui])
            expr = _format_expr(oc, oui, oup['deviation'], "\\eta^c")
            L.append(
                f"& {opt} {op_inner}_{{{vl} \\in \\mathcal{{U}}^c_{{bud}}}} \\{{ {expr} \\}} && \\text{{[Budgeted Uncertainty Objective]}} \\\\")

    L.append(f"& \\text{{subject to}} \\\\")


    for i in range(nc):
        ct, cp, ui = cts[i], cps[i], cui[i]
        ci = i + 1
        iq = _ineq(cp['direction'])
        if ct == 0:
            L.append(
                f"& {_nominal_expr(nv, cp['nominal_coeffs'])} {iq} {_fmt(cp['b'])} && \\text{{[Deterministic Constraint]}} \\\\")
        else:
            op_inner = "\\max" if cp['direction'] == 1 else ("\\min" if cp['direction'] == -1 else "\\text{eval}")
            if ct == 3:
                vl = ", ".join([f"\\delta^a_{{{ci},{j + 1}}}" for j in ui])
                expr = _format_con_expr(ci, cp['nominal_coeffs'], ui, None, "\\delta^a")
                L.append(
                    f"& {op_inner}_{{{vl} \\in \\mathcal{{U}}^a_{{{ci},poly}}}} \\{{ {expr} \\}} {iq} {_fmt(cp['b'])} && \\text{{[Polyhedral Uncertainty Constraint]}} \\\\")
            elif ct == 1:
                bounds = ", ".join([f"-1 \\le \\xi^a_{{{ci},{j + 1}}} \\le 1" for j in ui])
                expr = _format_con_expr(ci, cp['nominal_coeffs'], ui, cp['deviation'],
                                        f"\\xi^a_{{{ci}}}")
                expr = expr.replace(f"\\xi^a_{{{ci}}}_{{", f"\\xi^a_{{{ci},")
                L.append(
                    f"& {op_inner}_{{{bounds}}} \\{{ {expr} \\}} {iq} {_fmt(cp['b'])} && \\text{{[Box Uncertainty Constraint]}} \\\\")
            else:
                vl = ", ".join([f"\\eta^a_{{{ci},{j + 1}}}" for j in ui])
                expr = _format_con_expr(ci, cp['nominal_coeffs'], ui, cp['deviation'], f"\\eta^a_{{{ci}}}")
                expr = expr.replace(f"\\eta^a_{{{ci}}}_{{", f"\\eta^a_{{{ci},")
                L.append(
                    f"& {op_inner}_{{{vl} \\in \\mathcal{{U}}^a_{{{ci},bud}}}} \\{{ {expr} \\}} {iq} {_fmt(cp['b'])} && \\text{{[Budgeted Uncertainty Constraint]}} \\\\")

    has_sets = out in [2, 3] or any(ct in [2, 3] for ct in cts)
    if has_sets:
        L.append(f"& \\text{{where the uncertainty sets are defined as:}} \\\\")


    if out == 3:
        vl = ", ".join([f"\\delta^c_{{{j + 1}}}" for j in oui])
        sp = _poly_set_deviation_scalar([f"\\delta^c_{{{j + 1}}}" for j in oui], oup['poly_constraints'], pl, pu)
        L.append(f"& \\mathcal{{U}}^c_{{poly}} = \\{{ {vl} : {', '.join(sp)} \\}} \\\\")
    elif out == 2:
        vl = ", ".join([f"\\eta^c_{{{j + 1}}}" for j in oui])
        sp = [f"-1 \\le \\eta^c_{{{j + 1}}} \\le 1" for j in oui]
        sp_budget = " + ".join([f"|\\eta^c_{{{j + 1}}}|" for j in oui]) + f" \\le {_fmt(oup['gamma'])}"
        L.append(f"& \\mathcal{{U}}^c_{{bud}} = \\{{ {vl} : {', '.join(sp)}, {sp_budget} \\}} \\\\")

    for i in range(nc):
        ct, cp, ui = cts[i], cps[i], cui[i]
        if ct == 0 or ct == 1: continue
        ci = i + 1
        if ct == 3:
            vl = ", ".join([f"\\delta^a_{{{ci},{j + 1}}}" for j in ui])
            sp = _poly_set_deviation_scalar([f"\\delta^a_{{{ci},{j + 1}}}" for j in ui], cp['poly_constraints'], pl, pu)
            L.append(f"& \\mathcal{{U}}^a_{{{ci},poly}} = \\{{ {vl} : {', '.join(sp)} \\}} \\\\")
        elif ct == 2:
            vl = ", ".join([f"\\eta^a_{{{ci},{j + 1}}}" for j in ui])
            sp = [f"-1 \\le \\eta^a_{{{ci},{j + 1}}} \\le 1" for j in ui]
            sp_budget = " + ".join([f"|\\eta^a_{{{ci},{j + 1}}}|" for j in ui]) + f" \\le {_fmt(cp['gamma'])}"
            L.append(f"& \\mathcal{{U}}^a_{{{ci},bud}} = \\{{ {vl} : {', '.join(sp)}, {sp_budget} \\}} \\\\")

    bnd_true = ", ".join([_x_bnd_str(j, xl, xu) for j in range(nv)])
    L.append(f"& {bnd_true} \\\\")

    L.append("\\end{align*}")
    return "\n".join(L)


def _template_110(params):
    nv, nc, oc, od, out, oup, cts, cps, oui, cui, nr, pr, xl, xu, pl, pu = _extract_params(params)
    L = ["\\begin{align*}"]
    xvars_str = f"\\mathbf{{x}}"

    def _vec(v):
        return "[" + ", ".join([_fmt(x) for x in v]) + "]^\\top"

    def _mat(P):
        rows = []
        for row in P:
            rows.append(" & ".join([_fmt(x) for x in row]))
        return "\\begin{bmatrix} " + " \\\\ ".join(rows) + " \\end{bmatrix}"

    def _poly_mat_vec(ui, poly_constraints, pl_val, pu_val, nv):
        P_rows = []
        q_vals = []
        E_rows = []
        e_vals = []
        for A, b, direction in poly_constraints:
            row = [0] * nv
            for k, j in enumerate(ui):
                row[j] = A[k]
            if direction == 1:
                P_rows.append(row)
                q_vals.append(b)
            elif direction == -1:
                P_rows.append([-x for x in row])
                q_vals.append(-b)
            else:
                E_rows.append(row)
                e_vals.append(b)
        for j in ui:
            row = [0] * nv
            row[j] = 1
            P_rows.append(row)
            q_vals.append(pu_val)
            row_neg = [0] * nv
            row_neg[j] = -1
            P_rows.append(row_neg)
            q_vals.append(-pl_val)

        det_indices = [j for j in range(nv) if j not in ui]
        if det_indices:
            for j in det_indices:
                row_eq = [0] * nv
                row_eq[j] = 1
                E_rows.append(row_eq)
                e_vals.append(0)
            return _mat(P_rows), _vec(q_vals), _mat(E_rows), _vec(e_vals)
        if len(E_rows) > 0:
            return _mat(P_rows), _vec(q_vals), _mat(E_rows), _vec(e_vals)
        return _mat(P_rows), _vec(q_vals), None, None

    def _diag(devs, ui, nv):
        d = [0.0] * nv
        for k, j in enumerate(ui): d[j] = devs[k]
        return "\\text{diag}(" + ", ".join([_fmt(x) for x in d]) + ")"

    defs = []


    if out == 0:
        opt = "\\max_{" + xvars_str + "}" if od else "\\min_{" + xvars_str + "}"
        L.append(f"& {opt} \\mathbf{{c}}^\\top \\mathbf{{x}} \\\\")
        defs.append(f"\\mathbf{{c}} = {_vec(oc)}")
    else:
        opt = "\\max_{" + xvars_str + "}" if od else "\\min_{" + xvars_str + "}"
        op_inner = "\\min" if od else "\\max"
        cb = "\\boldsymbol{\\delta}_c"
        L.append(
            f"& {opt} {op_inner}_{{{cb} \\in \\mathcal{{U}}_c}} (\\mathbf{{c}}^{{nom}} + {cb})^\\top \\mathbf{{x}} \\\\")
        defs.append(f"\\mathbf{{c}}^{{nom}} = {_vec(oc)}")


    for i in range(nc):
        ct, cp = cts[i], cps[i]
        ci = i + 1
        iq = _ineq(cp['direction'])
        if ct == 0:
            L.append(f"& \\mathbf{{a}}_{{{ci}}}^\\top \\mathbf{{x}} {iq} b_{{{ci}}} \\\\")
            defs.append(f"\\mathbf{{a}}_{{{ci}}} = {_vec(cp['nominal_coeffs'])}")
            defs.append(f"b_{{{ci}}} = {_fmt(cp['b'])}")
        else:
            op_inner = "\\max" if cp['direction'] == 1 else ("\\min" if cp['direction'] == -1 else "\\text{eval}")
            eb = f"\\boldsymbol{{\\delta}}_{{a{ci}}}"
            L.append(
                f"& {op_inner}_{{{eb} \\in \\mathcal{{U}}_{{a{ci}}}}} (\\mathbf{{a}}_{{{ci}}}^{{nom}} + {eb})^\\top \\mathbf{{x}} {iq} b_{{{ci}}} \\\\")
            defs.append(f"\\mathbf{{a}}_{{{ci}}}^{{nom}} = {_vec(cp['nominal_coeffs'])}")
            defs.append(f"b_{{{ci}}} = {_fmt(cp['b'])}")


    if out > 0:
        cb = "\\boldsymbol{\\delta}_c"
        if out == 3:
            P_mat, q_vec, E_mat, e_vec = _poly_mat_vec(oui, oup['poly_constraints'], pl, pu, nv)
            if E_mat is not None:
                L.append(
                    f"& \\mathcal{{U}}_c = \\{{ {cb} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{D}}_c {cb} \\le \\mathbf{{d}}_c, \\quad \\mathbf{{E}}_c {cb} = \\mathbf{{e}}_c \\}} \\\\")
                defs.append(f"\\mathbf{{D}}_c = {P_mat}")
                defs.append(f"\\mathbf{{d}}_c = {q_vec}")
                defs.append(f"\\mathbf{{E}}_c = {E_mat}")
                defs.append(f"\\mathbf{{e}}_c = {e_vec}")
            else:
                L.append(
                    f"& \\mathcal{{U}}_c = \\{{ {cb} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{D}}_c {cb} \\le \\mathbf{{d}}_c \\}} \\\\")
                defs.append(f"\\mathbf{{D}}_c = {P_mat}")
                defs.append(f"\\mathbf{{d}}_c = {q_vec}")
        elif out == 1:
            dev_vec = [0.0] * nv
            for k, j in enumerate(oui): dev_vec[j] = oup['deviation'][k]
            L.append(
                f"& \\mathcal{{U}}_c = \\{{ {cb} \\in \\mathbb{{R}}^{{{nv}}} : |{cb}| \\le \\hat{{\\mathbf{{c}}}} \\}} \\\\")
            defs.append(f"\\hat{{\\mathbf{{c}}}} = {_vec(dev_vec)}")
        elif out == 2:
            dev_vec = [0.0] * nv
            for k, j in enumerate(oui): dev_vec[j] = oup['deviation'][k]
            L.append(
                f"& \\mathcal{{U}}_c = \\{{ {cb} \\in \\mathbb{{R}}^{{{nv}}} : |{cb}| \\le \\hat{{\\mathbf{{c}}}}, \\|\\text{{diag}}(\\hat{{\\mathbf{{c}}}})^{{-1}} {cb}\\|_1 \\le \\Gamma_c \\}} \\\\")
            defs.append(f"\\hat{{\\mathbf{{c}}}} = {_vec(dev_vec)}")
            defs.append(f"\\Gamma_c = {_fmt(oup['gamma'])}")

    for i in range(nc):
        ct, cp, ui = cts[i], cps[i], cui[i]
        if ct == 0: continue
        ci = i + 1
        eb = f"\\boldsymbol{{\\delta}}_{{a{ci}}}"
        if ct == 3:
            P_mat, q_vec, E_mat, e_vec = _poly_mat_vec(ui, cp['poly_constraints'], pl, pu, nv)
            if E_mat is not None:
                L.append(
                    f"& \\mathcal{{U}}_{{a{ci}}} = \\{{ {eb} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{D}}_{{a{ci}}} {eb} \\le \\mathbf{{d}}_{{a{ci}}}, \\quad \\mathbf{{E}}_{{a{ci}}} {eb} = \\mathbf{{e}}_{{a{ci}}} \\}} \\\\")
                defs.append(f"\\mathbf{{D}}_{{a{ci}}} = {P_mat}")
                defs.append(f"\\mathbf{{d}}_{{a{ci}}} = {q_vec}")
                defs.append(f"\\mathbf{{E}}_{{a{ci}}} = {E_mat}")
                defs.append(f"\\mathbf{{e}}_{{a{ci}}} = {e_vec}")
            else:
                L.append(
                    f"& \\mathcal{{U}}_{{a{ci}}} = \\{{ {eb} \\in \\mathbb{{R}}^{{{nv}}} : \\mathbf{{D}}_{{a{ci}}} {eb} \\le \\mathbf{{d}}_{{a{ci}}} \\}} \\\\")
                defs.append(f"\\mathbf{{D}}_{{a{ci}}} = {P_mat}")
                defs.append(f"\\mathbf{{d}}_{{a{ci}}} = {q_vec}")
        elif ct == 1:
            dev_vec = [0.0] * nv
            for k, j in enumerate(ui): dev_vec[j] = cp['deviation'][k]
            L.append(
                f"& \\mathcal{{U}}_{{a{ci}}} = \\{{ {eb} \\in \\mathbb{{R}}^{{{nv}}} : |{eb}| \\le \\hat{{\\mathbf{{a}}}}_{{{ci}}} \\}} \\\\")
            defs.append(f"\\hat{{\\mathbf{{a}}}}_{{{ci}}} = {_vec(dev_vec)}")
        elif ct == 2:
            dev_vec = [0.0] * nv
            for k, j in enumerate(ui): dev_vec[j] = cp['deviation'][k]
            L.append(
                f"& \\mathcal{{U}}_{{a{ci}}} = \\{{ {eb} \\in \\mathbb{{R}}^{{{nv}}} : |{eb}| \\le \\hat{{\\mathbf{{a}}}}_{{{ci}}}, \\|\\text{{diag}}(\\hat{{\\mathbf{{a}}}}_{{{ci}}})^{{-1}} {eb}\\|_1 \\le \\Gamma_{{a{ci}}} \\}} \\\\")
            defs.append(f"\\hat{{\\mathbf{{a}}}}_{{{ci}}} = {_vec(dev_vec)}")
            defs.append(f"\\Gamma_{{a{ci}}} = {_fmt(cp['gamma'])}")

    x_vec_str = "\\begin{bmatrix} " + " \\\\ ".join([f"x_{{{j + 1}}}" for j in range(nv)]) + " \\end{bmatrix}"
    defs.insert(0, f"\\mathbf{{x}} = {x_vec_str}")
    defs.insert(1, _x_bnd_compact(xl, xu, nv))

    for d in defs:
        L.append(f"& {d} \\\\")

    L.append("\\end{align*}")
    return "\n".join(L).replace("\\mathbb{R}^3_+", f"\\mathbb{{R}}^{{{nv}}}")


def _template_111(params):
    nv, nc, oc, od, out, oup, cts, cps, oui, cui, nr, pr, xl, xu, pl, pu = _extract_params(params)
    L = ["\\begin{align*}"]

    def _vec(v):
        return "[" + ", ".join([_fmt(x) for x in v]) + "]^\\top"

    def _mat(P):
        rows = []
        for row in P:
            rows.append(" & ".join([_fmt(x) for x in row]))
        return "\\begin{bmatrix} " + " \\\\ ".join(rows) + " \\end{bmatrix}"

    def _poly_mat_vec(ui, poly_constraints, pl_val, pu_val, nv):
        P_rows = []
        q_vals = []
        E_rows = []
        e_vals = []
        for A, b, direction in poly_constraints:
            row = [0] * nv
            for k, j in enumerate(ui):
                row[j] = A[k]
            if direction == 1:
                P_rows.append(row)
                q_vals.append(b)
            elif direction == -1:
                P_rows.append([-x for x in row])
                q_vals.append(-b)
            else:
                E_rows.append(row)
                e_vals.append(b)
        for j in ui:
            row = [0] * nv
            row[j] = 1
            P_rows.append(row)
            q_vals.append(pu_val)
            row_neg = [0] * nv
            row_neg[j] = -1
            P_rows.append(row_neg)
            q_vals.append(-pl_val)

        det_indices = [j for j in range(nv) if j not in ui]
        if det_indices:
            for j in det_indices:
                row_eq = [0] * nv
                row_eq[j] = 1
                E_rows.append(row_eq)
                e_vals.append(0)
            return _mat(P_rows), _vec(q_vals), _mat(E_rows), _vec(e_vals)
        if len(E_rows) > 0:
            return _mat(P_rows), _vec(q_vals), _mat(E_rows), _vec(e_vals)
        return _mat(P_rows), _vec(q_vals), None, None

    def _diag(devs, ui, nv):
        d = [0.0] * nv
        for k, j in enumerate(ui): d[j] = devs[k]
        return "\\text{diag}(" + ", ".join([_fmt(x) for x in d]) + ")"

    opt = "\\text{Maximize}" if od else "\\text{Minimize}"


    defs = {}
    defs['x_bnd'] = _x_bnd_compact(xl, xu, nv)


    if out == 0:
        L.append(f"& {opt} \\quad \\bar{{c}}^\\top x && \\text{{(Deterministic Objective)}} \\\\")
        defs['bar_c'] = f"\\bar{{c}} = {_vec(oc)}"
    else:
        op_inner = "\\min" if od else "\\max"
        lbl = {1: "(Box Uncertainty Objective)", 2: "(Budgeted Uncertainty Objective)",
               3: "(Polyhedral Uncertainty Objective)"}[out]
        if out == 3:
            cb = "\\delta_c"
            sn = "\\mathcal{U}_{c,P}"
            L.append(
                f"& {opt} \\quad {op_inner}_{{{cb} \\in {sn}}} ( \\bar{{c}} + {cb} )^\\top x && \\text{{{lbl}}} \\\\")
            defs['bar_c'] = f"\\bar{{c}} = {_vec(oc)}"
        elif out in [1, 2]:
            cb = "\\xi_c"
            sn = "\\mathcal{U}_{c,Box}" if out == 1 else "\\mathcal{U}_{c,Bud}"
            L.append(
                f"& {opt} \\quad {op_inner}_{{{cb} \\in {sn}}} \\left( \\bar{{c}} + \\text{{diag}}(\\hat{{c}}) {cb} \\right)^\\top x && \\text{{{lbl}}} \\\\")
            defs['bar_c'] = f"\\bar{{c}} = {_vec(oc)}"

            dev_vec = [0.0] * nv
            for k, j in enumerate(oui): dev_vec[j] = oup['deviation'][k]
            defs['hat_c'] = f"\\hat{{c}} = {_vec(dev_vec)}"

    L.append(f"& \\text{{subject to}} \\\\")


    for i in range(nc):
        ct, cp = cts[i], cps[i]
        ci = i + 1
        iq = _ineq(cp['direction'])
        if ct == 0:
            L.append(f"& a_{{{ci}}}^\\top x {iq} b_{{{ci}}} && \\text{{(Deterministic Constraint)}} \\\\")
            defs[f'a_{ci}'] = f"a_{{{ci}}} = {_vec(cp['nominal_coeffs'])}"
            defs[f'b_{ci}'] = f"b_{{{ci}}} = {_fmt(cp['b'])}"
        else:
            op = "\\max" if cp['direction'] == 1 else ("\\min" if cp['direction'] == -1 else "\\text{eval}")
            lbl = {1: "(Box Uncertainty Constraint)", 2: "(Budgeted Uncertainty Constraint)",
                   3: "(Polyhedral Uncertainty Constraint)"}[ct]
            if ct == 3:
                sn = f"\\mathcal{{U}}_{{a{ci},P}}"
                eb = f"\\delta_{{a{ci}}}"
                L.append(
                    f"& {op}_{{{eb} \\in {sn}}} ( \\bar{{a}}_{{{ci}}} + {eb} )^\\top x {iq} b_{{{ci}}} && \\text{{{lbl}}} \\\\")
                defs[f'bar_a_{ci}'] = f"\\bar{{a}}_{{{ci}}} = {_vec(cp['nominal_coeffs'])}"
                defs[f'b_{ci}'] = f"b_{{{ci}}} = {_fmt(cp['b'])}"
            elif ct in [1, 2]:
                sn = f"\\mathcal{{U}}_{{a{ci},Box}}" if ct == 1 else f"\\mathcal{{U}}_{{a{ci},Bud}}"
                eb = f"\\xi_{{a{ci}}}"
                L.append(
                    f"& {op}_{{{eb} \\in {sn}}} \\left( \\bar{{a}}_{{{ci}}} + \\text{{diag}}(\\hat{{a}}_{{{ci}}}) {eb} \\right)^\\top x {iq} b_{{{ci}}} && \\text{{{lbl}}} \\\\")
                defs[f'bar_a_{ci}'] = f"\\bar{{a}}_{{{ci}}} = {_vec(cp['nominal_coeffs'])}"
                defs[f'b_{ci}'] = f"b_{{{ci}}} = {_fmt(cp['b'])}"

                dev_vec = [0.0] * nv
                for k, j in enumerate(cui[i]): dev_vec[j] = cp['deviation'][k]
                defs[f'hat_a_{ci}'] = f"\\hat{{a}}_{{{ci}}} = {_vec(dev_vec)}"

    has_sets = out > 0 or any(ct > 0 for ct in cts)
    if has_sets:
        L.append(f"& \\text{{where:}} \\\\")


    if out == 3:
        P_mat, q_vec, E_mat, e_vec = _poly_mat_vec(oui, oup['poly_constraints'], pl, pu, nv)
        if E_mat is not None:
            L.append(
                f"& \\mathcal{{U}}_{{c,P}} = \\{{ \\delta_c \\in \\mathbb{{R}}^{{{nv}}} : D_c \\delta_c \\le d_c, \\quad E_c \\delta_c = e_c \\}} \\\\")
            defs['D_c'] = f"D_c = {P_mat}"
            defs['d_c'] = f"d_c = {q_vec}"
            defs['E_c'] = f"E_c = {E_mat}"
            defs['e_c'] = f"e_c = {e_vec}"
        else:
            L.append(
                f"& \\mathcal{{U}}_{{c,P}} = \\{{ \\delta_c \\in \\mathbb{{R}}^{{{nv}}} : D_c \\delta_c \\le d_c \\}} \\\\")
            defs['D_c'] = f"D_c = {P_mat}"
            defs['d_c'] = f"d_c = {q_vec}"
    elif out == 1:
        L.append(
            f"& \\mathcal{{U}}_{{c,Box}} = \\{{ \\xi_c \\in \\mathbb{{R}}^{{{nv}}} : \\|\\xi_c\\|_{{\\infty}} \\le 1 \\}} \\\\")
    elif out == 2:
        L.append(
            f"& \\mathcal{{U}}_{{c,Bud}} = \\{{ \\xi_c \\in \\mathbb{{R}}^{{{nv}}} : \\|\\xi_c\\|_{{\\infty}} \\le 1, \\|\\xi_c\\|_1 \\le \\Gamma_c \\}} \\\\")
        defs['Gamma_c'] = f"\\Gamma_c = {_fmt(oup['gamma'])}"

    for i in range(nc):
        ct, cp, ui = cts[i], cps[i], cui[i]
        if ct == 0: continue
        ci = i + 1
        if ct == 3:
            eb = f"\\delta_{{a{ci}}}"
            P_mat, q_vec, E_mat, e_vec = _poly_mat_vec(ui, cp['poly_constraints'], pl, pu, nv)
            if E_mat is not None:
                L.append(
                    f"& \\mathcal{{U}}_{{a{ci},P}} = \\{{ {eb} \\in \\mathbb{{R}}^{{{nv}}} : D_{{a{ci}}} {eb} \\le d_{{a{ci}}}, \\quad E_{{a{ci}}} {eb} = e_{{a{ci}}} \\}} \\\\")
                defs[f'D_a_{ci}'] = f"D_{{a{ci}}} = {P_mat}"
                defs[f'd_a_{ci}'] = f"d_{{a{ci}}} = {q_vec}"
                defs[f'E_a_{ci}'] = f"E_{{a{ci}}} = {E_mat}"
                defs[f'e_a_{ci}'] = f"e_{{a{ci}}} = {e_vec}"
            else:
                L.append(
                    f"& \\mathcal{{U}}_{{a{ci},P}} = \\{{ {eb} \\in \\mathbb{{R}}^{{{nv}}} : D_{{a{ci}}} {eb} \\le d_{{a{ci}}} \\}} \\\\")
                defs[f'D_a_{ci}'] = f"D_{{a{ci}}} = {P_mat}"
                defs[f'd_a_{ci}'] = f"d_{{a{ci}}} = {q_vec}"
        elif ct == 1:
            eb = f"\\xi_{{a{ci}}}"
            L.append(
                f"& \\mathcal{{U}}_{{a{ci},Box}} = \\{{ {eb} \\in \\mathbb{{R}}^{{{nv}}} : \\|{eb}\\|_{{\\infty}} \\le 1 \\}} \\\\")
        elif ct == 2:
            eb = f"\\xi_{{a{ci}}}"
            L.append(
                f"& \\mathcal{{U}}_{{a{ci},Bud}} = \\{{ {eb} \\in \\mathbb{{R}}^{{{nv}}} : \\|{eb}\\|_{{\\infty}} \\le 1, \\|{eb}\\|_1 \\le \\Gamma_{{a{ci}}} \\}} \\\\")
            defs[f'Gamma_a_{ci}'] = f"\\Gamma_{{a{ci}}} = {_fmt(cp['gamma'])}"

    x_vec_str = "[" + ", ".join([f"x_{{{j + 1}}}" for j in range(nv)]) + "]^\\top"
    L.append(f"& x = {x_vec_str} \\\\")


    for key, val in defs.items():
        L.append(f"& {val} \\\\")

    L.append("\\end{align*}")
    return "\n".join(L)


_TEMPLATE_MAP = {
    "000": _template_000,
    "001": _template_001,
    "010": _template_010,
    "011": _template_011,
    "100": _template_100,
    "101": _template_101,
    "110": _template_110,
    "111": _template_111,
}


def generate_robust_description(params, template_param):
    """Render a robust LP using three binary presentation choices: quantifier, expansion, and terminology."""
    key = f"{template_param[0]}{template_param[1]}{template_param[2]}"
    if key not in _TEMPLATE_MAP:
        raise ValueError(f"Unsupported template parameters: {template_param}")
    return _TEMPLATE_MAP[key](params)


def batch_generate_json(var_list, samples_per_n, template_list=None, seed=None, mode="all"):
    """Generate solver-verified records; mode selects all templates or one random template per instance."""
    if template_list is None:
        template_list = [[0, 0, 0]]

    if mode not in ("all", "random"):
        raise ValueError(f"mode must be 'all' or 'random', got: {mode!r}")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"Random seed: {seed}")

    all_data = []


    global_stats = {
        "C1": {"count": 0},
        "C2": {"count": 0},
    }
    global_robust_attempts = 0
    n_templates = len(template_list)

    for n_vars in var_list:
        mode_desc = f"{n_templates} templates per instance" if mode == "all" else "one randomly selected template"
        print(f"\nGenerating {n_vars}-variable instances, mode: {mode} ({mode_desc}); requested accepted instances: {samples_per_n} ===")
        valid_count = 0
        attempt_count = 0

        with tqdm(total=samples_per_n, desc=f"{n_vars}d") as pbar:
            while valid_count < samples_per_n:
                attempt_count += 1
                params = generate_random_params(n_vars)


                all_uncertainty_types = [params["objective_uncertainty_type"]] + params["constraint_types"]
                if all(ut == 0 for ut in all_uncertainty_types):
                    continue

                result = solve_robust_optimization(params)
                robust_attempts = result["robust_attempts"]
                global_robust_attempts += robust_attempts

                if result["status"] == "optimal" and not np.all(
                        np.isclose(result["solution"], params['x_lower']) |
                        np.isclose(result["solution"], params['x_upper'])):
                    category = "C1"
                    uncertainty_type_list = [params["objective_uncertainty_type"]] + params["constraint_types"]
                    solution = (
                        result["solution"].tolist()
                        if hasattr(result["solution"], "tolist")
                        else result["solution"]
                    )


                    if mode == "all":
                        selected_templates = template_list
                    else:
                        selected_templates = [random.choice(template_list)]

                    for template_param in selected_templates:
                        tp_str = ''.join(map(str, template_param))
                        desc = generate_robust_description(params, template_param=template_param)
                        record_id = f"{n_vars}_{valid_count + 1}_T{tp_str}"

                        record = {
                            "id": record_id,
                            "n_vars": n_vars,
                            "n_constraints": params["n_constraints"],
                            "template_param": template_param,
                            "description": desc,
                            "uncertainty_type": uncertainty_type_list,
                            "solution": solution,
                            "optimal_value": result["optimal_value"]
                        }
                        all_data.append(record)

                    valid_count += 1
                    pbar.update(1)
                else:
                    category = "C2"
                global_stats[category]["count"] += 1

        records_per_instance = n_templates if mode == "all" else 1
        print(
            f"  {n_vars}d: accepted {valid_count} instances x {records_per_instance} templates = {valid_count * records_per_instance} records (attempts: {attempt_count})")

    print("\nGeneration summary")
    for cat in ["C1", "C2"]:
        print(f"  {cat}: {global_stats[cat]['count']}")
    print(f"  Total solves: {global_robust_attempts}")
    print(f"  Total records: {len(all_data)}")
    print("=================================\n")

    return all_data


def main():
    """Generate fresh training examples from the command line."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vars", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--samples-per-n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--templates", nargs="+", default=["000", "001", "010", "011", "100", "101", "110", "111"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(n < 2 for n in args.vars) or args.samples_per_n < 1:
        parser.error("Dimensions must be at least 2 and samples-per-n must be positive.")
    if any(len(t) != 3 or set(t) - {"0", "1"} for t in args.templates):
        parser.error("Templates must be three binary digits, for example 000 or 111.")
    records = batch_generate_json(args.vars, args.samples_per_n,
        template_list=[[int(b) for b in t] for t in args.templates], seed=args.seed, mode="random")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
