"""MPS parsing, robust-counterpart construction, and LaTeX rendering."""
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

def dec(token):
    return Decimal(token.replace('D', 'E').replace('d', 'e'))

def parse(path):
    section = None
    row_types = {}
    row_order = []
    col_order = []
    coeffs = defaultdict(lambda: defaultdict(Decimal))
    rhs = defaultdict(Decimal)
    bound_records = []
    range_records = []
    for raw in path.read_text(encoding='ascii', errors='replace').splitlines():
        if not raw.strip() or raw.lstrip().startswith('*'):
            continue
        parts = raw.split()
        head = parts[0].upper()
        if not raw[0].isspace() and head in {'NAME', 'ROWS', 'COLUMNS', 'RHS', 'RANGES', 'BOUNDS', 'ENDATA'}:
            section = head
            continue
        if section == 'ROWS':
            typ, row = (parts[0].upper(), parts[1])
            row_types[row] = typ
            row_order.append(row)
        elif section == 'COLUMNS':
            if len(parts) >= 3 and parts[1].strip("'").upper() != 'MARKER':
                col = parts[0]
                if col not in coeffs:
                    col_order.append(col)
                for i in range(1, len(parts) - 1, 2):
                    coeffs[col][parts[i]] += dec(parts[i + 1])
        elif section == 'RHS':
            for i in range(1, len(parts) - 1, 2):
                rhs[parts[i]] = dec(parts[i + 1])
        elif section == 'RANGES':
            range_records.append(parts)
        elif section == 'BOUNDS':
            bound_records.append(parts)
    if range_records:
        raise NotImplementedError('RANGES are not supported by this counterpart builder')
    obj_row = next((row for row in row_order if row_types[row] == 'N'))
    return (row_types, row_order, col_order, coeffs, rhs, bound_records, obj_row)

def fmt(value):
    value = Decimal(value)
    if value == 0:
        return '0'
    return format(value, 'f').rstrip('0').rstrip('.') if '.' in format(value, 'f') else format(value, 'f')

def expr(terms):
    out = []
    for coef, name in terms:
        coef = Decimal(coef)
        if coef == 0:
            continue
        sign = '+' if coef > 0 else '-'
        mag = abs(coef)
        atom = name if mag == 1 else f'{fmt(mag)} {name}'
        if not out:
            out.append(atom if sign == '+' else f'- {atom}')
        else:
            out.append(f' {sign} {atom}')
    return ''.join(out) or '0'

def build_counterpart(src, eps, delta, dest):
    row_types, row_order, cols, coeffs, rhs, bound_records, obj = parse(src)
    lb = {col: Decimal(0) for col in cols}
    ub = {col: None for col in cols}
    for rec in bound_records:
        typ = rec[0].upper()
        col = rec[2]
        val = dec(rec[3]) if len(rec) > 3 else None
        if typ == 'LO':
            lb[col] = val
        elif typ == 'UP':
            ub[col] = val
        elif typ == 'FX':
            lb[col] = ub[col] = val
        elif typ == 'FR':
            lb[col] = ub[col] = None
        elif typ == 'MI':
            lb[col] = None
        elif typ == 'PL':
            ub[col] = None
        elif typ == 'BV':
            lb[col], ub[col] = (Decimal(0), Decimal(1))
        elif typ == 'LI':
            lb[col] = val
        elif typ == 'UI':
            ub[col] = val
        else:
            raise NotImplementedError(f'bound type {typ}')
    fixed = {col: lb[col] for col in cols if lb[col] is not None and lb[col] == ub[col]}
    active_cols = [col for col in cols if col not in fixed]
    xid = {col: f'x{i}' for i, col in enumerate(active_cols)}
    rhs_eff = defaultdict(Decimal, rhs)
    for row in row_order:
        rhs_eff[row] = rhs[row] - sum((coeffs[col].get(row, Decimal(0)) * value for col, value in fixed.items()), Decimal(0))
    row_terms = defaultdict(list)
    uncertain_cols = set()
    uncertain_terms = defaultdict(list)
    for col in active_cols:
        for row, coef in coeffs[col].items():
            if row == obj:
                continue
            row_terms[row].append((coef, xid[col]))
            if row_types.get(row) in {'L', 'G'} and Fraction(coef).denominator > 100:
                uncertain_cols.add(col)
                uncertain_terms[row].append((abs(coef), col))
    yid = {col: f'y{i}' for i, col in enumerate(cols) if col in uncertain_cols}
    lines = ['Minimize']
    lines.append(' obj: ' + expr(((coeffs[col].get(obj, 0), xid[col]) for col in active_cols)))
    lines.append('Subject To')
    cnum = 0
    for row in row_order:
        typ = row_types[row]
        if typ == 'N':
            continue
        sense = {'L': '<=', 'G': '>=', 'E': '='}[typ]
        lines.append(f' n{cnum}: {expr(row_terms[row])} {sense} {fmt(rhs_eff[row])}')
        cnum += 1
    eps = Decimal(str(eps))
    delta = Decimal(str(delta))
    for row in row_order:
        typ = row_types[row]
        if typ not in {'L', 'G'} or not uncertain_terms[row]:
            continue
        orient = Decimal(1) if typ == 'L' else Decimal(-1)
        terms = [(orient * coef, name) for coef, name in row_terms[row]]
        terms += [(eps * coef, yid[col]) for coef, col in uncertain_terms[row]]
        brhs = orient * rhs_eff[row] + delta * max(Decimal(1), abs(rhs_eff[row]))
        lines.append(f' r{cnum}: {expr(terms)} <= {fmt(brhs)}')
        cnum += 1
    for col in active_cols:
        if col in uncertain_cols:
            lines.append(f' a{cnum}: {yid[col]} - {xid[col]} >= 0')
            cnum += 1
            lines.append(f' a{cnum}: {yid[col]} + {xid[col]} >= 0')
            cnum += 1
    lines.append('Bounds')
    for col in active_cols:
        x = xid[col]
        lo, hi = (lb[col], ub[col])
        if lo is None and hi is None:
            lines.append(f' {x} free')
        elif lo is not None and hi is not None and (lo == hi):
            lines.append(f' {x} = {fmt(lo)}')
        elif lo is None:
            lines.append(f' {x} <= {fmt(hi)}')
        elif hi is None:
            if lo != 0:
                lines.append(f' {fmt(lo)} <= {x}')
        else:
            lines.append(f' {fmt(lo)} <= {x} <= {fmt(hi)}')
    lines.append('End')
    dest.write_text('\n'.join(lines) + '\n', encoding='ascii')
    print(f'built {dest} rows={cnum} active_vars={len(active_cols)} fixed_substituted={len(fixed)} abs_vars={len(yid)}')

def atom(coef, variable):
    mag = abs(coef)
    return variable if mag == 1 else f'{fmt(mag)}{variable}'

def deterministic_expr(terms):
    parts = []
    for coef, variable in terms:
        if not coef:
            continue
        sign = '+' if coef > 0 else '-'
        body = atom(coef, variable)
        parts.append((sign, body))
    if not parts:
        return '0'
    first_sign, first_body = parts[0]
    out = ('-' if first_sign == '-' else '') + first_body
    for sign, body in parts[1:]:
        out += f' {sign} {body}'
    return out

def uncertain_expr(terms, ugly_cols, eps, row_index):
    parts = []
    for coef, col_index, variable in terms:
        if not coef:
            continue
        if col_index in ugly_cols:
            nominal = fmt(coef)
            radius = fmt(eps * abs(coef))
            body = f'({nominal} + {radius}\\xi_{{{row_index},{col_index}}}){variable}'
            sign = '+'
        else:
            sign = '+' if coef > 0 else '-'
            body = atom(coef, variable)
        parts.append((sign, body))
    if not parts:
        return '0'
    first_sign, first_body = parts[0]
    out = ('-' if first_sign == '-' else '') + first_body
    for sign, body in parts[1:]:
        out += f' {sign} {body}'
    return out

def render_description(src, eps, delta):
    row_types, row_order, cols, coeffs, rhs, bound_records, obj = parse(src)
    x = {col: f'x_{{{j + 1}}}' for j, col in enumerate(cols)}
    lb = {col: Decimal(0) for col in cols}
    ub = {col: None for col in cols}
    for rec in bound_records:
        typ, col = (rec[0].upper(), rec[2])
        val = dec(rec[3]) if len(rec) > 3 else None
        if typ == 'LO':
            lb[col] = val
        elif typ == 'UP':
            ub[col] = val
        elif typ == 'FX':
            lb[col] = ub[col] = val
        elif typ == 'FR':
            lb[col] = ub[col] = None
        elif typ == 'MI':
            lb[col] = None
        elif typ == 'PL':
            ub[col] = None
        else:
            raise NotImplementedError(typ)
    rows = defaultdict(list)
    for j, col in enumerate(cols, 1):
        for row, coef in coeffs[col].items():
            if row != obj:
                rows[row].append((coef, j, x[col]))
    lines = ['\\begin{align*}']
    objective = deterministic_expr(((coeffs[col].get(obj, 0), x[col]) for col in cols))
    lines.append(f'& \\text{{minimize }} {objective} \\\\')
    lines.append('& \\text{subject to} \\\\')
    row_idx = 0
    for row in row_order:
        typ = row_types[row]
        if typ == 'N':
            continue
        row_idx += 1
        orient = Decimal(-1) if typ == 'G' else Decimal(1)
        terms = [(orient * c, j, var) for c, j, var in rows[row]]
        target = orient * rhs[row]
        if typ == 'E':
            body = deterministic_expr(((c, var) for c, _, var in terms))
            lines.append(f'& {body} = {fmt(target)} \\\\[-1mm]')
            continue
        nominal = deterministic_expr(((c, var) for c, _, var in terms))
        lines.append(f'& {nominal} \\le {fmt(target)} \\\\[-1mm]')
        ugly_cols = {j for c, j, _ in terms if Fraction(c).denominator > 100}
        if ugly_cols:
            body = uncertain_expr(terms, ugly_cols, eps, row_idx)
            xis = ', '.join((f'-1\\le\\xi_{{{row_idx},{j}}}\\le1' for j in sorted(ugly_cols)))
            relaxed = target + delta * max(Decimal(1), abs(target))
            lines.append(f'& \\max_{{{xis}}}\\left\\{{{body}\\right\\}} \\le {fmt(relaxed)} \\\\[-1mm]')
    nonnegative = [j for j, col in enumerate(cols, 1) if lb[col] == 0 and ub[col] is None]
    free = [j for j, col in enumerate(cols, 1) if lb[col] is None and ub[col] is None]
    if nonnegative:
        ids = ','.join(map(str, nonnegative))
        lines.append(f'& x_j\\ge0\\quad\\forall j\\in\\{{{ids}\\}} \\\\[-1mm]')
    if free:
        ids = ','.join(map(str, free))
        lines.append(f'& x_j\\in\\mathbb{{R}}\\quad\\forall j\\in\\{{{ids}\\}} \\\\[-1mm]')
    for j, col in enumerate(cols, 1):
        lo, hi = (lb[col], ub[col])
        var = x[col]
        if lo == 0 and hi is None or (lo is None and hi is None):
            continue
        if lo is not None and hi is not None and (lo == hi):
            lines.append(f'& {var}={fmt(lo)} \\\\[-1mm]')
        elif lo is None:
            lines.append(f'& {var}\\le {fmt(hi)} \\\\[-1mm]')
        elif hi is None:
            lines.append(f'& {fmt(lo)}\\le {var} \\\\[-1mm]')
        else:
            lines.append(f'& {fmt(lo)}\\le {var}\\le {fmt(hi)} \\\\[-1mm]')
    lines.append('\\end{align*}')
    return '\n'.join(lines)
