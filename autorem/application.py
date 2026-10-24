"""Natural-language application formulation and robust reformulation."""
from __future__ import annotations

import json
import time

from .roles import Reformulator
from .llm import LLMAPIError, LLMNetworkError

_RF_ROLE_AND_TASK = """\
You are a Reformulator.

Your task is to convert the input robust optimization problem into a tractable robust counterpart — a fully explicit deterministic formulation in which all effects of uncertainty are resolved, with no shorthand operators remaining (e.g., absolute values |·|, max/min, norms, or supremum expressions).

Use the experience memory to assist: match each minimal independent reformulation unit (e.g., a single uncertain constraint or uncertain objective term) to the most relevant experience if one exists, and apply that experience's reformulation approach; derive the counterpart from scratch for unmatched units.

All specific numeric values from the problem instance must be preserved and appear explicitly in the final answer.

### CRITICAL INSTRUCTION ON NUMERIC PRECISION:
- All nominal values AND perturbation/uncertainty values MUST be preserved exactly as given in the problem — do not omit or alter any of them
- When numeric values require computation (e.g., combining parameters, square roots, fractional exponents, or division), ALWAYS retain the exact analytic expression in your final latex output to avoid precision loss\
"""


_RF_OUTPUT_FORMAT = """\
==================================================================
OUTPUT FORMAT
==================================================================

Your ENTIRE response MUST be a single valid JSON object — nothing else.
  • Do NOT include any text, commentary, or markdown outside the JSON.
  • Do NOT wrap the JSON in ``` code fences.
  • The FIRST character of your response MUST be `{{` and the LAST MUST be `}}`.

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


def make_application_reformulator_prompt(experiences: str, question: str) -> str:
    if not experiences or experiences.strip() == "":
        experiences = "empty"

    input_block = (
        "==================================================================\n"
        "INPUT\n"
        "==================================================================\n\n"
        f"=== PROBLEM ===\n{question}\n\n"
        f"=== REFORMULATION EXPERIENCE MEMORY ===\n{experiences}"
    )

    return "\n\n".join([_RF_ROLE_AND_TASK, input_block, _RF_OUTPUT_FORMAT])


class ApplicationReformulator(Reformulator):
    """Reformulate application models while retaining exact analytic expressions."""
    def generate(self, *, rem, question):
        from autorem.roles import ReformulatorOutput, _safe_call
        import json as _json
        all_ids = rem.list_experience_ids()
        prompt = make_application_reformulator_prompt(
            experiences=rem.render_experiences(ids=all_ids),
            question=question,
        )
        resp = _safe_call(self.client.complete, prompt, allow_truncation_passthrough=True)

        # ---- Token-limit truncation ----
        if getattr(resp, '_is_truncated', False):
            ct = getattr(resp, 'completion_tokens', 0) or 0
            raw = getattr(resp, 'text', "") or ""
            return ReformulatorOutput(
                reasoning=(
                    f"[TRUNCATION] Reformulator output was cut off because it exceeded the token output limit "
                    f"(finish_reason='length', completion_tokens={ct})."
                ),
                final_answer=raw, raw={},
                latency=getattr(resp, 'latency', 0.0) or 0.0,
                prompt_tokens=getattr(resp, 'prompt_tokens', 0) or 0,
                completion_tokens=ct,
                matched_experience_ids=[], is_truncated=True, raw_text=raw,
            )

        data = _json.loads(resp.text)
        is_json_failed = bool(data.get("_json_failed", False))

        if is_json_failed:
            raw_text = data.get("final_answer", "") or data.get("rewritten_problem", "")
            return ReformulatorOutput(
                reasoning=str(data.get("reasoning", "")),
                final_answer=str(raw_text), raw=data,
                latency=getattr(resp, 'latency', 0.0) or 0.0,
                prompt_tokens=getattr(resp, 'prompt_tokens', 0) or 0,
                completion_tokens=getattr(resp, 'completion_tokens', 0) or 0,
                matched_experience_ids=[], is_json_failed=True, raw_text=str(raw_text),
            )

        raw_ids = data.get("matched_experience_ids", [])
        if not isinstance(raw_ids, list):
            raw_ids = []
        valid_ids = set(rem.list_experience_ids())
        matched_ids = [int(i) for i in raw_ids if isinstance(i, (int, float)) and int(i) in valid_ids]

        return ReformulatorOutput(
            reasoning=str(data.get("reasoning", "")),
            final_answer=str(data.get("final_answer", "")),
            raw=data,
            latency=resp.latency,
            prompt_tokens=getattr(resp, "prompt_tokens", None),
            completion_tokens=getattr(resp, "completion_tokens", None),
            matched_experience_ids=matched_ids,
            raw_text=resp.text,
        )


def make_formulator_prompt(question: str) -> str:
    instruction = r"""
You are a Formulator.

Your task is to formulate the given natural language problem description into a precise mathematical optimization model.

### CRITICAL INSTRUCTION ON ROBUST OPTIMIZATION:
If the problem involves robust optimization or uncertainty, you MUST output the ORIGINAL robust formulation exactly as described — including all uncertainty sets, their structures, and how they appear in the constraints or objective.

- DO NOT derive the Robust Counterpart (RC) or any deterministic equivalent.
- DO NOT apply duality, minimax reformulation, or any algebraic transformation to eliminate the uncertainty.
- DO NOT simplify or merge uncertainty sets.

These reformulation and derivation steps will be handled by a dedicated downstream **Reformulator** module that specializes in converting the original robust model into its tractable counterpart. Your role is only to faithfully transcribe the problem into a clean mathematical model — preserving all uncertain parameters and their uncertainty descriptions verbatim.

The output MUST be a complete mathematical formulation in LaTeX format, including:
- Decision variables with clear definitions
- Objective function (minimize or maximize)
- All constraints (nominal and robust/uncertain ones)
- Uncertainty set descriptions (if applicable)
- Variable bounds and domains
- All numeric values from the problem instance appearing explicitly, including nominal values AND perturbation/uncertainty values
- All nominal values AND perturbation/uncertainty values MUST be preserved exactly as given in the problem — do not omit or alter any of them
- When numeric values require computation (e.g., combining parameters, square roots, fractional exponents, or division), ALWAYS retain the exact analytic expression in your final latex output to avoid precision loss
"""

    body = f"""
=== PROBLEM ===
{question}

Return ONLY a RAW JSON object (no markdown, no extra text):
{{
  "reasoning": <Step-by-step thinking process of how you extract and formulate the mathematical model from the problem description>,
  "final_answer": <The complete mathematical formulation in LaTeX format>
}}
Do not wrap the JSON in ``` or any other markdown. The first character MUST be `{{` and the last MUST be `}}`.
"""
    return instruction.strip() + "\n\n" + body.strip()


def formulate(client, description: str) -> dict:
    """Formulate a natural-language application and return its output and usage."""
    prompt = make_formulator_prompt(description)
    for attempt in range(3):
        try:
            resp = client.complete(prompt)
            text = resp.text.strip()

            if text.startswith("```json"):
                text = text[len("```json"):].strip()
            if text.startswith("```"):
                text = text[3:].strip()
            if text.endswith("```"):
                text = text[:-3].strip()

            data = json.loads(text)
            return {
                "raw": data,
                "reasoning": data.get("reasoning", ""),
                "final_answer": data.get("final_answer", ""),
                "latency": resp.latency,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
            }
        except json.JSONDecodeError:
            try:
                fixed = text
                if fixed.count('"') % 2 == 1:
                    fixed += '"'
                if fixed.count("{") > fixed.count("}"):
                    fixed += "}" * (fixed.count("{") - fixed.count("}"))
                data = json.loads(fixed)
                return {
                    "raw": data,
                    "reasoning": data.get("reasoning", ""),
                    "final_answer": data.get("final_answer", ""),
                    "latency": resp.latency,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                }
            except Exception:
                pass
            print(f"[Warning] JSON parse failed (attempt {attempt + 1}/3)")
        except (LLMNetworkError, LLMAPIError):
            raise
        except Exception as e:
            print(f"[Warning] LLM call failed (attempt {attempt + 1}/3): {e}")
            time.sleep(5)

    return {
        "raw": {},
        "reasoning": "",
        "final_answer": "",
        "latency": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
