# ====================================================================
# JSON FORMAT RULES — single shared module inserted into every prompt
# ====================================================================

_JSON_FORMAT_RULES = """\
Your ENTIRE response MUST be a single valid JSON object — nothing else.
  • Do NOT include any text, commentary, or markdown outside the JSON.
  • Do NOT wrap the JSON in ``` code fences.
  • The FIRST character of your response MUST be `{` and the LAST MUST be `}`.\
"""


def _inject_json_format_rules(template: str) -> str:
    """Replace the {json_format_rules} placeholder in an output-format template."""
    return template.replace("{json_format_rules}", _JSON_FORMAT_RULES)


def _content_status_label(base_label: str, status: str) -> str:
    """
    Return an annotated section label for downstream prompt building
    when the content is truncated or could not be parsed as JSON.
    """
    if status == "truncated":
        return (
            f"{base_label} "
            f"(⚠ TRUNCATED — this is the raw incomplete output, NOT the properly formatted content)"
        )
    elif status == "json_failed":
        return (
            f"{base_label} "
            f"(⚠ JSON PARSE FAILED — this is the raw malformed output, NOT the properly formatted content)"
        )
    return base_label


# ====================================================================
# REFORMULATOR
# ====================================================================

_RF_ROLE_AND_TASK = """\
You are a Reformulator.

Your task is to convert the input robust optimization problem into a tractable robust counterpart — a fully explicit deterministic formulation in which all effects of uncertainty are resolved, with no shorthand operators remaining (e.g., absolute values |·|, max/min, norms, or supremum expressions).

Use the experience memory to assist: match each minimal independent reformulation unit (e.g., a single uncertain constraint or uncertain objective term) to the most relevant experience if one exists, and apply that experience's reformulation approach; derive the counterpart from scratch for unmatched units.

All specific numeric values from the problem instance must be preserved and appear explicitly in the final answer.\
"""

_RF_OUTPUT_FORMAT = """\
==================================================================
OUTPUT FORMAT
==================================================================

{json_format_rules}

"reasoning" MUST cover:
  (a) For each reformulation unit: which experience was matched (or none) and why.
  (b) The derivation process that transforms each unit into its tractable robust counterpart.

"matched_experience_ids" MUST be a list of integer experience IDs matched to reformulation units in (a). Use [] if no unit was matched.

"final_answer" MUST be a complete mathematical formulation of the tractable robust counterpart in LaTeX within a \\begin{{align}}...\\end{{align}} environment, with all numeric values from the problem instance appearing explicitly.

Return ONLY a RAW JSON object with the following structure:
{{
  "reasoning": <matching analysis and derivation process for each reformulation unit>,
  "matched_experience_ids": [<list of matched experience IDs as integers>],
  "final_answer": <The tractable robust counterpart with numeric values in LaTeX format>
}}\
"""


def make_reformulator_prompt(experiences: str, question: str) -> str:
    if not experiences or experiences.strip() == "":
        experiences = "empty"

    input_block = (
        "==================================================================\n"
        "INPUT\n"
        "==================================================================\n\n"
        f"=== PROBLEM ===\n{question}\n\n"
        f"=== REFORMULATION EXPERIENCE MEMORY ===\n{experiences}"
    )

    prompt = "\n\n".join([
        _RF_ROLE_AND_TASK,
        input_block,
        _inject_json_format_rules(_RF_OUTPUT_FORMAT),
    ])
    return prompt


# ====================================================================
# CODER
# ====================================================================

_CD_ROLE_AND_TASK = """\
You are a Coder.

Your task is to write Python code that uses the `gurobipy` library to solve the given optimization problem (a tractable robust counterpart).

Import gurobipy and create a Model for the given problem. Solve the model, and then print the optimal objective value using print(model.ObjVal). If the model is infeasible or unbounded, print "infeasible" or "unbounded" accordingly.

Note: In Gurobi, the default lower bound for variables is 0. Free variables (unbounded) MUST be declared explicitly with a negative infinity lower bound (e.g., lb=float('-inf') or lb=-GRB.INFINITY).\
"""

_CD_OUTPUT_FORMAT = """\
==================================================================
OUTPUT FORMAT
==================================================================

{json_format_rules}

Return ONLY a RAW JSON object with the following structure:
{{
  "reasoning": <Step-by-step thinking process of how you translate the formulation into code>,
  "final_answer": <The complete Python code>
}}\
"""


def make_coder_prompt(latex_formulation: str, formulation_status: str = "ok") -> str:
    formulation_label = _content_status_label("PROBLEM (ROBUST COUNTERPART)", formulation_status)
    input_block = (
        "==================================================================\n"
        "INPUT\n"
        "==================================================================\n\n"
        f"=== {formulation_label} ===\n{latex_formulation}"
    )

    return "\n\n".join([
        _CD_ROLE_AND_TASK,
        input_block,
        _inject_json_format_rules(_CD_OUTPUT_FORMAT),
    ])


# ====================================================================
# REFLECTOR
# ====================================================================

_R_ROLE_AND_TASK = """\
You are a Reflector.

Your task is to reflect on all training samples in the current step (both correct and incorrect) and update the reformulation experience memory — a collection of experiences that each capture how a specific type of robust optimization structure should be correctly reformulated into its tractable robust counterpart. Use three operators: `add`, `update`, and `delete`. Each experience_id MUST appear at most once across all operators.

The usage scores shown in the memory (success / failure counts) record how often each experience has led to a correct or incorrect final answer when used.

Note: A problem may match multiple experiences (one per reformulation unit). A failure does not imply every matched experience is wrong — the root cause may lie in only one of the matched experiences, while others are correct.

In your `reasoning`, analyze the root cause of each failure by examining the Reformulator's matched experience IDs, its reasoning, and the environment feedback for all samples. Use the correct samples as reference to understand what works, and focus your operator decisions on resolving the failures without disrupting what is already correct. Consider all samples together and decide on a unified set of memory operations. Then decide which operators to apply:
  - `add` a new experience if no suitable one exists in memory.
  - `update` an existing experience if its content is incorrect or misleading.
  - `delete` an experience if it is harmful.\
"""

_R_OUTPUT_FORMAT = """\
==================================================================
OUTPUT FORMAT
==================================================================

{json_format_rules}

Fields per operator:
- add:    {{"operator": "add", "experience_id": int, "content": "..."}}
- update: {{"operator": "update", "experience_id": int, "content": "..."}}
- delete: {{"operator": "delete", "experience_id": int}}

Note on `add` experience_id: assign `max(existing experience ids) + 1` for the first `add`,
`max(existing experience ids) + 2` for the second, and so on. Each `add` MUST use a distinct new ID.

Return ONLY a RAW JSON object with the following structure:
{{
  "reasoning": <step-by-step thinking covering: (1) failure diagnosis, (2) correct reformulation derivation for the affected unit(s), and (3) operator decisions>,
  "final_answer": [
    {{ ... }}
  ]
}}\
"""


def make_reflector_prompt(
    questions: list[str],
    full_memory: str,
    reformulator_raw_outputs: list[str],
    environment_feedbacks: list[str],
    ems: list[int],
    previous_reflector_trace: str | None = None,
    reformulator_output_statuses: list[str] | None = None,
    double_check_failures: list[dict] | None = None,
) -> str:
    """
    Build the Reflector prompt for all samples in the step (correct and incorrect).

    full_memory: rendered memory showing all experiences with content and quality scores.
    reformulator_raw_outputs: full raw JSON text from the Reformulator for each sample.
    ems: list of 0/1 correctness indicators per sample (1=correct, 0=incorrect).
    reformulator_output_statuses: list of "ok" | "truncated" | "json_failed" per sample.
    double_check_failures: list of dicts describing validation samples that failed after the
        previous memory update (only provided when the training sample passed but DC failed).
        Each dict has keys: source ("linked"|"unlinked"), question, reformulator_raw_output,
        environment_feedback, em.
    """
    if not full_memory or full_memory.strip() == "":
        full_memory = "empty"

    n = len(questions)
    if reformulator_output_statuses is None:
        reformulator_output_statuses = ["ok"] * n

    # Build per-sample blocks
    sample_blocks = []
    for idx, (question, raw_output, feedback, status, em) in enumerate(
        zip(questions, reformulator_raw_outputs, environment_feedbacks,
            reformulator_output_statuses, ems)
    ):
        if status != "ok":
            refm_label = _content_status_label("REFORMULATOR'S RAW RESPONSE", status)
            reformulator_block = (
                f"=== {refm_label} ===\n"
                f"[NOTE: The Reformulator's response could not be properly parsed. "
                f"The complete raw response is shown below.]\n\n"
                f"{raw_output}"
            )
        else:
            reformulator_block = f"=== REFORMULATOR'S OUTPUT ===\n{raw_output}"

        verdict = "CORRECT" if em == 1 else "INCORRECT"
        block = (
            f"--- SAMPLE {idx + 1} ({verdict}) ---\n\n"
            f"=== PROBLEM ===\n{question}\n\n"
            f"{reformulator_block}\n\n"
            f"=== ENVIRONMENT FEEDBACK ===\n{feedback}"
        )
        sample_blocks.append(block)

    samples_section = "\n\n".join(sample_blocks)

    input_block = (
        "==================================================================\n"
        "INPUT\n"
        "==================================================================\n\n"
        f"=== EXISTING REFORMULATION EXPERIENCE MEMORY ===\n{full_memory}\n\n"
        f"{samples_section}"
    )

    previous_trace_block = ""
    if previous_reflector_trace is not None:
        previous_trace_block = (
            "\n\n==================================================================\n"
            "PREVIOUS ITERATION TRACE\n"
            "==================================================================\n\n"
            "The following is your reasoning and actions in the previous inner iteration, "
            "provided for reference only.\n\n"
            "The existing reformulation experience memory has already been updated to reflect "
            "the effects of these actions.\n\n"
            f"=== PREVIOUS REFLECTOR TRACE (REASONING + OPERATORS) ===\n"
            f"{previous_reflector_trace}\n\n"
            "You must analyze why the previous reasoning and actions did not resolve the failure, "
            "and decide what different decision, if any, should be made now."
        )

    dc_block = ""
    if double_check_failures:
        dc_sample_parts = []
        for i, dc in enumerate(double_check_failures):
            source_label = (
                "LINKED (previously correct using the modified experience)"
                if dc.get("source") == "linked"
                else "UNLINKED (no prior association with the modified experience)"
            )
            dc_refm_out = dc.get("reformulator_raw_output", "(unavailable)")
            dc_sample_parts.append(
                f"--- DC SAMPLE {i + 1} [source: {source_label}] ---\n\n"
                f"=== PROBLEM ===\n{dc.get('question', '')}\n\n"
                f"=== REFORMULATOR'S OUTPUT ===\n{dc_refm_out}\n\n"
                f"=== ENVIRONMENT FEEDBACK ===\n{dc.get('environment_feedback', '')}"
            )
        dc_block = (
            "\n\n==================================================================\n"
            "DOUBLE CHECK FAILURES\n"
            "==================================================================\n\n"
            "After your previous memory update, the training sample was solved correctly.\n"
            "However, when the updated memory was tested on the following validation samples, "
            "some of them FAILED — indicating that the memory change caused regression on "
            "previously correct (or unrelated) problems.\n\n"
            "LINKED samples were previously solved correctly using the modified experience. "
            "UNLINKED samples had no prior association with the modified experience.\n\n"
            "You MUST revise your memory operations so that both the training sample AND "
            "these validation samples are handled correctly.\n\n"
            + "\n\n".join(dc_sample_parts)
        )

    prompt = "\n\n".join([
        _R_ROLE_AND_TASK,
        input_block + previous_trace_block + dc_block,
        _inject_json_format_rules(_R_OUTPUT_FORMAT),
    ])
    return prompt

