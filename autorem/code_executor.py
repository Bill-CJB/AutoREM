"""
Gurobi Code Executor.

Executes a Python code string that uses gurobipy.
Returns (objective_value, solution, error_type, error_message).
"""
from __future__ import annotations
import json
import math
import re
import tempfile
import subprocess
import sys
import os

_SOLUTION_MARKER = "__SOLUTION_JSON__"


def _inject_solution_extraction(code: str) -> str:
    """
    Inject solution-extraction code right after ``print(model.ObjVal)``
    (or any variant like ``m.ObjVal``).

    The injected snippet:
      - lives in the **same scope** as the model variable
      - collects variables whose names start with 'x' (the original
        decision variables, not dual / auxiliary ones)
      - prints a tagged JSON line that the caller can parse
      - is wrapped in try/except so it never breaks the original code
    """
    # Match  print(model.ObjVal)  /  print( m.ObjVal )  etc.
    pattern = r'([ \t]*)(print\s*\(\s*(\w+)\.ObjVal\s*\))'

    def _replacement(m):
        indent = m.group(1)
        original = m.group(2)
        model_var = m.group(3)
        snippet = (
            f"\n{indent}try:\n"
            f"{indent}    import json as __json\n"
            f"{indent}    __xvars = {{v.VarName: v.X for v in {model_var}.getVars() if v.VarName.lower().startswith('x')}}\n"
            f'{indent}    print("{_SOLUTION_MARKER}:" + __json.dumps(__xvars))\n'
            f"{indent}except Exception:\n"
            f"{indent}    pass"
        )
        return indent + original + snippet

    modified = re.sub(pattern, _replacement, code)
    return modified


def _parse_solution_from_stdout(stdout: str) -> dict | None:
    """
    Scan *stdout* for the tagged solution line and return a dict,
    or ``None`` if not found / unparseable.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(_SOLUTION_MARKER + ":"):
            json_str = line[len(_SOLUTION_MARKER) + 1:]
            try:
                sol = json.loads(json_str)
                if isinstance(sol, dict):
                    return sol
            except Exception:
                pass
    return None


def execute_gurobi_code(code_string: str, timeout: int = 60):
    """
    Execute a Python code string that uses gurobipy.

    Returns
    -------
    tuple of (float | None, dict | None, str | None, str | None)
        (objective_value, solution, error_type, error_message)
        - Success:        (float_value, {var: val, ...}, None, None)
        - Code error:     (None, None, "code_error", detail)
        - Parse error:    (None, None, "parse_error", detail)
        - Infeasible:     (None, None, "infeasible", detail)
        - Unbounded:      (None, None, "unbounded", detail)

    ``solution`` contains only the original decision variables (names
    starting with 'x').  It is ``None`` when extraction fails.
    """
    if not isinstance(code_string, str) or code_string.strip() == "":
        return None, None, "code_error", "Empty or non-string code input."
    if timeout <= 0:
        return None, None, "code_error", "Execution timeout must be positive."

    # --- inject solution extraction ---
    augmented_code = _inject_solution_extraction(code_string)

    try:
        # This is process isolation, not a security sandbox. Run generated code
        # only in a suitably isolated environment for untrusted inputs.
        child_env = {
            key: value for key, value in os.environ.items()
            if not key.upper().endswith(("API_KEY", "API_TOKEN", "ACCESS_TOKEN", "SECRET_KEY"))
        }
        with tempfile.TemporaryDirectory(prefix="autorem_solver_") as work_dir:
            code_path = os.path.join(work_dir, "solve.py")
            with open(code_path, "w", encoding="utf-8") as handle:
                handle.write(augmented_code)
            result = subprocess.run(
                [sys.executable, "-I", code_path],
                capture_output=True, text=True, timeout=timeout,
                cwd=work_dir, env=child_env, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Truncate very long error messages
            if len(stderr) > 500:
                stderr = stderr[:500] + "..."
            return None, None, "code_error", f"Code execution failed:\n{stderr}"

        stdout = result.stdout.strip()
        if not stdout:
            return None, None, "parse_error", "Code produced no output."

        # --- extract solution (best-effort) ---
        solution = _parse_solution_from_stdout(stdout)

        # --- strip the injected marker line so it doesn't confuse obj parsing ---
        clean_lines = [
            ln for ln in stdout.splitlines()
            if not ln.strip().startswith(_SOLUTION_MARKER + ":")
        ]
        clean_stdout = "\n".join(clean_lines).strip()

        # Check for infeasible/unbounded
        last_line = clean_stdout.split('\n')[-1].strip() if clean_stdout else ""
        if "infeasible" in last_line.lower() or "infeasible" in clean_stdout.lower():
            return None, None, "infeasible", "Model is infeasible."
        if "unbounded" in last_line.lower() or "unbounded" in clean_stdout.lower():
            return None, None, "unbounded", "Model is unbounded."

        # Parse the last line as a float
        try:
            obj_val = float(last_line)
            if not math.isfinite(obj_val):
                return None, None, "parse_error", "Objective value must be finite."
            return obj_val, solution, None, None
        except ValueError:
            return None, None, "parse_error", f"Could not parse output as float: '{last_line}'"

    except subprocess.TimeoutExpired:
        return None, None, "code_error", f"Code execution timed out ({timeout}s)."
    except Exception as e:
        return None, None, "code_error", f"Unexpected error: {e}"
