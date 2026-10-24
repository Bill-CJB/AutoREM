from autorem.code_executor import execute_gurobi_code


class RobustTaskEnvironment:
    """Evaluate a robust counterpart by executing Gurobi Python code."""

    def evaluate(self, sample: dict, code_answer: str):
        """
        Execute the Gurobi code and compare with the ground truth.

        Parameters
        ----------
        sample : dict
            Must contain 'optimal_value' (ground truth).
        code_answer : str
            Python code string that uses gurobipy to solve the problem.

        Returns
        -------
        (int, str, str | None) : (em_score, feedback, error_type)
            error_type is "code_error", "parse_error", "infeasible", "unbounded", or None
        """
        gt = sample.get("optimal_value")
        gt_solution = sample.get("solution")

        obj_val, solution, error_type, error_msg = execute_gurobi_code(code_answer)

        # --- Code error (empty input, execution failure, timeout) ---
        if error_type == "code_error":
            fb = (
                f"Ground truth optimal objective value: {gt}\n\n"
                "Code execution error: the Gurobi Python code could not be executed successfully.\n\n"
                f"Error detail:\n{error_msg}"
            )
            return 0, fb, "code_error"

        # --- Parse error (no output, unparseable output) ---
        if error_type == "parse_error":
            fb = (
                f"Ground truth optimal objective value: {gt}\n\n"
                "Parse error: the code executed but the output could not be parsed as a numeric value.\n\n"
                f"Error detail:\n{error_msg}"
            )
            return 0, fb, "parse_error"

        # --- Infeasible ---
        if error_type == "infeasible":
            fb = (
                f"Ground truth optimal objective value: {gt}\n"
                f"Solver result: infeasible\n"
                "The result is incorrect.\n"
                "The code executed successfully, but the model is infeasible "
                "(the reformulated problem may be ill-formed).\n"
            )
            return 0, fb, "infeasible"

        # --- Unbounded ---
        if error_type == "unbounded":
            fb = (
                f"Ground truth optimal objective value: {gt}\n"
                f"Solver result: unbounded\n"
                "The result is incorrect.\n"
                "The code executed successfully, but the model is unbounded "
                "(the reformulated problem may be ill-formed).\n"
            )
            return 0, fb, "unbounded"

        # --- Solved successfully, but no value returned (should not happen) ---
        if obj_val is None:
            fb = (
                f"Ground truth optimal objective value: {gt}\n\n"
                "The solver completed but returned no objective value."
            )
            return 0, fb, None

        # --- Build solution feedback snippet ---
        solution_fb = ""
        if solution is not None:
            solution_fb = f"\nComputed decision variable values: {solution}\n"
        if gt_solution is not None:
            solution_fb += f"Ground truth decision variable values: {gt_solution}\n"

        # --- Check correctness ---
        if abs(obj_val - gt) < 1e-4:
            fb = (
                f"Ground truth optimal objective value: {gt}\n"
                f"Solver result: {obj_val}\n"
                "Result matches the ground truth."
                f"{solution_fb}"
            )
            return 1, fb, None

        # --- Wrong result ---
        fb = (
            f"Ground truth optimal objective value: {gt}\n"
            f"Solver result: {obj_val}\n"
            "The result is incorrect.\n"
            f"{solution_fb}"
        )
        return 0, fb, None
