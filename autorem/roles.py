from __future__ import annotations
import json
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from .memory import ReformulationExperienceMemory
from .llm import LLMAPIError, LLMNetworkError
from .prompts import (
    make_reformulator_prompt,
    make_coder_prompt,
    make_reflector_prompt,
)

# ------------------ JSON response helpers ------------------

_VALID_JSON_ESCAPE_CHARS = set('"\\/bfnrtu')

def _fix_invalid_json_escapes(text: str) -> str:
    """
    Fix invalid JSON escape sequences in LLM output.

    When the LLM embeds LaTeX in JSON strings it sometimes writes single
    backslashes (e.g. ``\\delta``, ``\\quad``) instead of the JSON-required
    double backslash (``\\\\delta``, ``\\\\quad``).  JSON only allows
    ``\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX`` as escape sequences;
    everything else (``\\d``, ``\\q``, ``\\m``, ``\\a``, ``\\s``, …) raises a
    JSONDecodeError.

    This function scans the raw text character-by-character:
    - When it sees ``\\`` followed by a *valid* JSON escape character it keeps
      both characters as-is (the escape is already correct).
    - When it sees ``\\`` followed by an *invalid* escape character it emits
      ``\\\\`` instead, turning the single backslash into a properly escaped
      backslash so that the LaTeX command survives JSON parsing intact.
    """
    result: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '\\' and i + 1 < n:
            next_c = text[i + 1]
            if next_c in _VALID_JSON_ESCAPE_CHARS:
                # Valid JSON escape – keep both chars and skip ahead
                result.append(c)
                result.append(next_c)
                i += 2
            else:
                # Invalid escape – double the lone backslash; the following
                # character will be picked up as a literal on the next iteration
                result.append('\\\\')
                i += 1
        else:
            result.append(c)
            i += 1
    return ''.join(result)


def _call_with_network_retry(func, *args, **kwargs):
    """Call func, retrying up to 12 times on LLMNetworkError (10 s between retries)."""
    last_exc = None
    for net_retry in range(12):
        try:
            return func(*args, **kwargs)
        except LLMNetworkError as e:
            last_exc = e
            print(f"[Warning] Network error ({net_retry + 1}/12): {e}")
            time.sleep(10)
    print("[Error] 12 consecutive network errors. Interrupting.")
    raise last_exc


def _safe_call(func, *args, allow_truncation_passthrough: bool = False, **kwargs):
    """
    Call func(*args, **kwargs) and ensure the response contains valid JSON.

    Behaviour:
    - finish_reason='length' (token-limit truncation):
        Attempts quick salvage; if salvage fails and allow_truncation_passthrough=True,
        returns immediately with _is_truncated=True (no retry).
    - Non-truncated JSON parse failure:
        Retries the full LLM call up to 3 times.
        After all retries, attempts JSON repair (_fix_invalid_json_escapes + bracket patching).
        If repair also fails, embeds the raw text in a fallback JSON dict and sets _json_failed=True.
    - Network errors:
        Retried up to 12 times (via _call_with_network_retry) before propagating.

    allow_truncation_passthrough: when True (offline training mode), a length-truncated
      response is returned as-is with _is_truncated=True so callers can handle it specially.
    """
    last_raw_text: str = ""
    last_result = None

    for attempt in range(3):
        try:
            result = _call_with_network_retry(func, *args, **kwargs)
            last_result = result
            raw_text = getattr(result, 'text', '') or ''
            last_raw_text = raw_text

            if getattr(result, 'finish_reason', None) == 'length':
                # Token-limit truncation
                salvage = raw_text
                try:
                    if not isinstance(json.loads(salvage), dict):
                        raise ValueError("The response must be a JSON object.")
                    return result  # valid JSON despite truncation
                except Exception:
                    # Try quick bracket patch
                    for closing in ['"}', '"]}', ']}', '}']:
                        try:
                            if not isinstance(json.loads(salvage + closing), dict):
                                continue
                            result.text = salvage + closing
                            return result
                        except Exception:
                            pass
                if allow_truncation_passthrough:
                    result._is_truncated = True
                    return result
                # Fall through to retry
                continue

            if not isinstance(json.loads(raw_text), dict):
                raise ValueError("The response must be a JSON object.")
            return result

        except (LLMNetworkError, LLMAPIError):
            raise
        except Exception:
            pass

    # All retries exhausted — attempt JSON repair
    if last_raw_text:
        try:
            fixed = _fix_invalid_json_escapes(last_raw_text)
            if not isinstance(json.loads(fixed), dict):
                raise ValueError("The response must be a JSON object.")
            last_result.text = fixed
            return last_result
        except Exception:
            pass
        # Try bracket patching on the repaired text
        try:
            fixed = _fix_invalid_json_escapes(last_raw_text)
            for closing in ['"}', '"]}', ']}', '}']:
                try:
                    if not isinstance(json.loads(fixed + closing), dict):
                        continue
                    last_result.text = fixed + closing
                    return last_result
                except Exception:
                    pass
        except Exception:
            pass

    # Final fallback: embed raw text, set _json_failed flag
    if last_result is not None:
        try:
            last_result._json_failed = True
            last_result.text = json.dumps({
                "reasoning": (
                    "[JSON_PARSE_FAILED] The LLM response could not be parsed as valid JSON "
                    "after 3 retries and a repair attempt. "
                    "The raw output is preserved in 'final_answer' and 'rewritten_problem'."
                ),
                "final_answer": last_raw_text,
                "rewritten_problem": last_raw_text,
                "_json_failed": True,
            })
            return last_result
        except Exception:
            pass

    # Absolute fallback
    fake = types.SimpleNamespace(
        text=json.dumps({
            "reasoning": "[JSON_PARSE_FAILED] No valid LLM response was received.",
            "final_answer": "",
            "rewritten_problem": "",
            "_json_failed": True,
        }),
        finish_reason="error",
        latency=0.0,
        prompt_tokens=0,
        completion_tokens=0,
    )
    fake._json_failed = True
    return fake


# ------------------ Data Classes ------------------

@dataclass
class ReformulatorOutput:
    reasoning: str
    final_answer: str
    raw: Dict[str, Any]
    latency: float
    prompt_tokens: int
    completion_tokens: int
    matched_experience_ids: List[int] = field(default_factory=list)
    is_truncated: bool = False
    is_json_failed: bool = False
    raw_text: str = ""


@dataclass
class CoderOutput:
    reasoning: str
    final_answer: str  # Gurobi Python code
    raw: Dict[str, Any]
    latency: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class ReflectorOutput:
    reasoning: str
    final_answer: List[Dict[str, Any]]
    raw: Dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0


# ------------------ Base LLM Client ------------------
class LLMClientProtocol:
    def complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError


# ------------------ Reformulator ------------------
class Reformulator:
    def __init__(self, client: LLMClientProtocol):
        self.client = client

    def generate(self, *, rem: ReformulationExperienceMemory, question: str) -> ReformulatorOutput:
        all_ids = rem.list_experience_ids()
        prompt = make_reformulator_prompt(
            experiences=rem.render_experiences(ids=all_ids),
            question=question,
        )
        resp = _safe_call(self.client.complete, prompt, allow_truncation_passthrough=True)

        # ---- Token-limit truncation: propagate raw text, skip JSON parsing ----
        if getattr(resp, '_is_truncated', False):
            ct = getattr(resp, 'completion_tokens', 0) or 0
            raw = getattr(resp, 'text', "") or ""
            return ReformulatorOutput(
                reasoning=(
                    f"[TRUNCATION] Reformulator output was cut off because it exceeded the token output limit "
                    f"(finish_reason='length', completion_tokens={ct})."
                ),
                final_answer=raw,
                raw={},
                latency=getattr(resp, 'latency', 0.0) or 0.0,
                prompt_tokens=getattr(resp, 'prompt_tokens', 0) or 0,
                completion_tokens=ct,
                matched_experience_ids=[],
                is_truncated=True,
                raw_text=raw,
            )

        data = json.loads(resp.text)
        is_json_failed = bool(data.get("_json_failed", False))

        if is_json_failed:
            raw_text = data.get("final_answer", "") or data.get("rewritten_problem", "")
            return ReformulatorOutput(
                reasoning=str(data.get("reasoning", "")),
                final_answer=str(raw_text),
                raw=data,
                latency=getattr(resp, 'latency', 0.0) or 0.0,
                prompt_tokens=getattr(resp, 'prompt_tokens', 0) or 0,
                completion_tokens=getattr(resp, 'completion_tokens', 0) or 0,
                matched_experience_ids=[],
                is_json_failed=True,
                raw_text=str(raw_text),
            )

        # Parse and validate matched_experience_ids
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


# ------------------ Coder ------------------
class Coder:
    def __init__(self, client: LLMClientProtocol):
        self.client = client

    def generate(self, *, latex_formulation: str, formulation_status: str = "ok") -> CoderOutput:
        prompt = make_coder_prompt(latex_formulation=latex_formulation, formulation_status=formulation_status)
        resp = _safe_call(self.client.complete, prompt)
        data = json.loads(resp.text)
        return CoderOutput(
            reasoning=str(data.get("reasoning", "")),
            final_answer=str(data.get("final_answer", "")),
            raw=data,
            latency=resp.latency,
            prompt_tokens=getattr(resp, "prompt_tokens", None),
            completion_tokens=getattr(resp, "completion_tokens", None),
        )


# ------------------ Reflector ------------------
class Reflector:
    """Single Reflector: handles unified content edits in one LLM call."""

    def __init__(self, client: LLMClientProtocol):
        self.client = client

    def generate(
        self,
        *,
        questions: List[str],
        rem: ReformulationExperienceMemory,
        reformulator_outputs: List[ReformulatorOutput],
        environment_feedbacks: List[str],
        ems: List[int],
        previous_reflector_trace: Optional[str] = None,
        double_check_failures: Optional[List[Dict[str, Any]]] = None,
    ) -> ReflectorOutput:
        # Show full memory catalog (content + quality scores) to the Reflector
        catalog_with_scores = rem.render_catalog(show_scores=True)
        full_memory_block = catalog_with_scores if catalog_with_scores.strip() else ""

        statuses = []
        for ro in reformulator_outputs:
            if ro.is_truncated:
                statuses.append("truncated")
            elif ro.is_json_failed:
                statuses.append("json_failed")
            else:
                statuses.append("ok")

        prompt = make_reflector_prompt(
            questions=questions,
            full_memory=full_memory_block,
            reformulator_raw_outputs=[ro.raw_text for ro in reformulator_outputs],
            environment_feedbacks=environment_feedbacks,
            ems=ems,
            previous_reflector_trace=previous_reflector_trace,
            reformulator_output_statuses=statuses,
            double_check_failures=double_check_failures,
        )
        resp = _safe_call(self.client.complete, prompt)
        data = json.loads(resp.text)
        return ReflectorOutput(
            reasoning=str(data.get("reasoning", "")),
            final_answer=data.get("final_answer", []),
            raw=data,
            prompt_tokens=getattr(resp, "prompt_tokens", None) or 0,
            completion_tokens=getattr(resp, "completion_tokens", None) or 0,
            latency=getattr(resp, "latency", None) or 0.0,
        )
