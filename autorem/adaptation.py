"""
adaptation.py
----------------------------
OfflineAdapter using a single Reflector with Experience-Gated Double Check (DC).

Training loop design:
  - N Epochs × 8 Steps per Epoch × 1 sample per Step
  - Each step:
      1. Evaluate training sample; skip if correct.
      2. If wrong, enter inner reflection loop (up to MAX_INNER_ITERS=3):
         a. Reflect → apply ops to working_rem
         b. Re-evaluate training sample on working_rem
         c. If still wrong → continue inner loop (train trace only, no DC info)
         d. If now correct → run Double Check on up to DC_SAMPLE_SIZE val samples
            - If all DC pass  → commit working_rem, update exp_val_correct_map, break
            - If any DC fail  → add DC failure info as separate module, continue inner loop
      3. If inner loop exhausted without commit → discard working_rem (no change)
  - Epoch end: run full val set, rebuild exp_val_correct_map, apply val gate.
"""
from __future__ import annotations
import contextlib
import io
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple, Set

from autorem.memory import ReformulationExperienceMemory
from autorem.llm import LLMNetworkError
from autorem.roles import (
    Reformulator,
    Coder,
    Reflector,
    ReformulatorOutput,
    CoderOutput,
    ReflectorOutput,
)
from generation.synthetic import batch_generate_json


# ── Training hyper-parameters ────────────────────────────────────────────────
STEPS_PER_EPOCH = 8
MAX_INNER_ITERS = 3


# ── Helpers ──────────────────────────────────────────────────────────────────
def _extract_content_from_op(op: dict) -> Optional[str]:
    """Extract unified content from a reflector op (uses the 'content' field)."""
    content = op.get("content", None)
    return str(content) if content is not None else None



# ============================================================
# Offline Adapter
# ============================================================

class OfflineAdapter:
    def __init__(
        self,
        *,
        rem: ReformulationExperienceMemory,
        reformulator: Reformulator,
        coder: Coder,
        reflector: Reflector,
        save_dir: str,
        val_data: str,
        epochs: int = 10,
        batch_size: int = 4,
        resume_from: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> None:
        self.rem = rem
        self.reformulator = reformulator
        self.coder = coder
        self.reflector = reflector
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_data = val_data
        self._counter_lock = threading.Lock()

        # ── Val sample cache (loaded once) ────────────────────────────
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive integers.")
        self.val_samples = json.loads(Path(val_data).read_text(encoding="utf-8"))
        if not isinstance(self.val_samples, list) or not self.val_samples:
            raise ValueError("Validation data must be a nonempty JSON list.")
        if not all(isinstance(sample, dict) and "description" in sample and "optimal_value" in sample for sample in self.val_samples):
            raise ValueError("Each validation sample must contain description and optimal_value.")
        if resume_from and not Path(resume_from).is_file():
            raise FileNotFoundError(resume_from)

        # ── Experience → val sample tracking ─────────────────────────
        # Maps experience_id → set of val sample indices (0-based) that were
        # correct the last time they were evaluated using that experience.
        self.exp_val_correct_map: Dict[int, Set[int]] = {}
        # Maps val sample index → latest em result (0 or 1).
        self.val_sample_results: Dict[int, int] = {}

        _timestamp = time.strftime('%Y%m%d_%H%M%S')
        _folder = f"{tag}_{_timestamp}" if tag else _timestamp
        self.save_dir = Path(f"{save_dir}/{_folder}")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ---------- log files ----------
        self.train_log = open(self.save_dir / "train.log", "a", encoding="utf-8")
        self.step_log = open(self.save_dir / "steps_detail.log", "a", encoding="utf-8")
        self.failed_log = open(self.save_dir / "failed_samples.log", "a", encoding="utf-8")
        self.metric_log = open(self.save_dir / "metrics.log", "a", encoding="utf-8")
        self.metric_log.write(
            "epoch,train_acc_init,train_acc_final,train_correct_init,train_correct_final,"
            "train_total,code_errors,rem_size,"
            "refm_prompt_tokens,refm_completion_tokens,refm_latency,"
            "coder_prompt_tokens,coder_completion_tokens,coder_latency,"
            "refl_prompt_tokens,refl_completion_tokens,refl_latency,"
            "train_fc_prompt_tokens,train_fc_completion_tokens,train_fc_latency,"
            "val_fc_prompt_tokens,val_fc_completion_tokens,val_fc_latency,"
            "dc_fc_prompt_tokens,dc_fc_completion_tokens,dc_fc_latency\n"
        )

        self.train_start_time: Optional[float] = None
        self.total_inner_iterations: int = 0
        self.total_reach_max_inner_iters: int = 0
        self.total_dc_checks: int = 0
        self.total_dc_passed: int = 0
        self.total_dc_failed: int = 0

        # Cumulative token/latency counters
        self.total_refm_prompt_tokens: int = 0
        self.total_refm_completion_tokens: int = 0
        self.total_refm_latency: float = 0.0
        self.total_coder_prompt_tokens: int = 0
        self.total_coder_completion_tokens: int = 0
        self.total_coder_latency: float = 0.0
        self.total_refl_prompt_tokens: int = 0
        self.total_refl_completion_tokens: int = 0
        self.total_refl_latency: float = 0.0

        # Split counters: training / val / double-check
        self.train_fc_prompt_tokens: int = 0
        self.train_fc_completion_tokens: int = 0
        self.train_fc_latency: float = 0.0
        self.val_fc_prompt_tokens: int = 0
        self.val_fc_completion_tokens: int = 0
        self.val_fc_latency: float = 0.0
        self.dc_fc_prompt_tokens: int = 0
        self.dc_fc_completion_tokens: int = 0
        self.dc_fc_latency: float = 0.0

        # Token-routing mode: "train" | "val" | "dc"
        self._token_mode: str = "train"

        self.start_epoch = 1
        self._resume_dir: Optional[Path] = None
        self._resume_best_val_acc: float = 0.0
        if resume_from and Path(resume_from).exists():
            payload = json.loads(Path(resume_from).read_text(encoding="utf-8"))
            self.rem = ReformulationExperienceMemory.from_dict(payload)
            self._resume_best_val_acc = float(payload.get("best_val_acc", 0.0))
            raw_map = payload.get("exp_val_correct_map", {})
            self.exp_val_correct_map = {
                int(k): set(int(x) for x in v) for k, v in raw_map.items()
            }
            raw_vsr = payload.get("val_sample_results", {})
            self.val_sample_results = {int(k): int(v) for k, v in raw_vsr.items()}
            name = Path(resume_from).stem
            if "epoch" in name:
                try:
                    self.start_epoch = int(name.split("epoch")[-1]) + 1
                except Exception:
                    pass
            self._resume_dir = Path(resume_from).parent
            print(
                f"[Resume] Loaded REM from {resume_from}, start epoch={self.start_epoch}, "
                f"best_val_acc={self._resume_best_val_acc:.3f}, "
                f"exp_val_correct_map entries={len(self.exp_val_correct_map)}, "
                f"val_sample_results entries={len(self.val_sample_results)}"
            )
            self._log(self.train_log,
                f"[Resume] Loaded REM from {resume_from}, start epoch={self.start_epoch}, "
                f"best_val_acc={self._resume_best_val_acc:.3f}")

    # ---------- logging helper ----------
    def _log(self, f, msg: str):
        f.write(msg + "\n")
        f.flush()

    # ---------- run one sample through the pipeline ----------
    def _run_pipeline_single(
        self, sample: Dict[str, Any], rem: ReformulationExperienceMemory, env,
    ) -> Tuple[int, str, Optional[str], ReformulatorOutput, CoderOutput]:
        """Returns (em, feedback, error_type, refm_out, coder_out)."""
        question = str(sample["description"])

        # 1) Reformulator
        refm_out: ReformulatorOutput = self.reformulator.generate(rem=rem, question=question)
        _refm_pt = refm_out.prompt_tokens or 0
        _refm_ct = refm_out.completion_tokens or 0
        _refm_lat = refm_out.latency or 0.0
        with self._counter_lock:
            self.total_refm_prompt_tokens += _refm_pt
            self.total_refm_completion_tokens += _refm_ct
            self.total_refm_latency += _refm_lat
            mode = self._token_mode
            if mode == "val":
                self.val_fc_prompt_tokens += _refm_pt
                self.val_fc_completion_tokens += _refm_ct
                self.val_fc_latency += _refm_lat
            elif mode == "dc":
                self.dc_fc_prompt_tokens += _refm_pt
                self.dc_fc_completion_tokens += _refm_ct
                self.dc_fc_latency += _refm_lat
            else:
                self.train_fc_prompt_tokens += _refm_pt
                self.train_fc_completion_tokens += _refm_ct
                self.train_fc_latency += _refm_lat

        # 2) Coder
        refm_status = (
            "truncated" if refm_out.is_truncated else
            "json_failed" if refm_out.is_json_failed else
            "ok"
        )
        coder_out: CoderOutput = self.coder.generate(
            latex_formulation=refm_out.final_answer,
            formulation_status=refm_status,
        )
        _coder_pt = coder_out.prompt_tokens or 0
        _coder_ct = coder_out.completion_tokens or 0
        _coder_lat = coder_out.latency or 0.0
        with self._counter_lock:
            self.total_coder_prompt_tokens += _coder_pt
            self.total_coder_completion_tokens += _coder_ct
            self.total_coder_latency += _coder_lat
            mode = self._token_mode
            if mode == "val":
                self.val_fc_prompt_tokens += _coder_pt
                self.val_fc_completion_tokens += _coder_ct
                self.val_fc_latency += _coder_lat
            elif mode == "dc":
                self.dc_fc_prompt_tokens += _coder_pt
                self.dc_fc_completion_tokens += _coder_ct
                self.dc_fc_latency += _coder_lat
            else:
                self.train_fc_prompt_tokens += _coder_pt
                self.train_fc_completion_tokens += _coder_ct
                self.train_fc_latency += _coder_lat

        # 3) Ground-truth evaluation
        em, feedback, error_type = env.evaluate(sample, coder_out.final_answer)
        return em, feedback, error_type, refm_out, coder_out

    # ---------- evaluate a batch in parallel ----------
    def _evaluate_batch(
        self, batch: List[Dict[str, Any]], rem: ReformulationExperienceMemory, env
    ) -> List[Tuple]:
        n = len(batch)
        results: List[Optional[Tuple]] = [None] * n
        print(f"    [Batch] Launching {n} threads...")
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = {
                executor.submit(self._run_pipeline_single, batch[i], rem, env): i
                for i in range(n)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    # ---------- apply reflector ops to working_rem ----------
    def _apply_reflector_ops(
        self,
        ops: List[Dict[str, Any]],
        working_rem: ReformulationExperienceMemory,
    ) -> List[int]:
        """Apply ops to working_rem. Returns list of deleted experience IDs.
        NOTE: does NOT touch self.exp_val_correct_map — caller handles that at commit time."""
        if not isinstance(ops, list):
            return []

        deleted_eids: List[int] = []

        add_ops = [op for op in ops if isinstance(op, dict) and op.get("operator") == "add"]
        update_ops = [op for op in ops if isinstance(op, dict) and op.get("operator") == "update"]
        delete_ops = [op for op in ops if isinstance(op, dict) and op.get("operator") == "delete"]

        for idx, _op in enumerate(add_ops):
            _expected_id = working_rem.peek_next_id() + idx
            _proposed_id = _op.get("experience_id")
            if _proposed_id != _expected_id:
                self._log(self.step_log,
                    f"[HardCtrl] Corrected add op experience_id: {_proposed_id} → {_expected_id}")
                _op["experience_id"] = _expected_id

        for op in add_ops:
            content = _extract_content_from_op(op)
            if content is None:
                self._log(self.step_log, f"[Drop] add op missing content: {op}")
                continue
            new_id = working_rem.add_experience(
                content=content, experience_id=op.get("experience_id"))
            print(f"[REM] Added experience_id={new_id}")
            self._log(self.step_log, f"[REM] add new experience_id={new_id}")

        for op in update_ops:
            eid = op.get("experience_id", None)
            content = _extract_content_from_op(op)
            if eid is None or content is None:
                self._log(self.step_log,
                    f"[Drop] update op missing experience_id or content: {op}")
                continue
            working_rem.update_content(int(eid), content)
            print(f"[REM] Updated experience_id={eid}")
            self._log(self.step_log, f"[REM] update content id={eid}")

        for op in delete_ops:
            eid = op.get("experience_id", None)
            if eid is None:
                continue
            working_rem.remove_experience(int(eid))
            deleted_eids.append(int(eid))
            print(f"[REM] Deleted experience_id={eid}")
            self._log(self.step_log, f"[REM] delete experience_id={eid}")

        return deleted_eids

    # ---------- select DC samples (called once per step, then locked) ----------
    def _select_dc_samples(
        self,
        changed_eids: List[int],
        step_idx: int,
        epoch: int,
    ) -> Tuple[List[int], Dict[int, str]]:
        """
        Select up to batch_size val samples for double-check.
        Called ONCE per step when training sample first becomes correct.
        The returned indices are then locked and reused for all subsequent DC calls in the step.

        Priority:
          (a) linked   – val indices in exp_val_correct_map for any changed experience
          (b) unlinked – all remaining val indices

        Returns: (selected_indices, source_map {idx: "linked"|"unlinked"})
        """
        total_val = len(self.val_samples)
        if total_val == 0:
            return [], {}

        linked_pool: Set[int] = set()
        for eid in changed_eids:
            linked_pool |= self.exp_val_correct_map.get(eid, set())
        # Only include val samples that are currently correct in unlinked pool;
        # samples already wrong should NOT be DC candidates.
        currently_correct = {
            i for i in range(total_val)
            if self.val_sample_results.get(i, 0) == 1
        }
        unlinked_pool = currently_correct - linked_pool

        n_linked = len(linked_pool)
        n_unlinked = len(unlinked_pool)

        selected_linked = random.sample(list(linked_pool), min(self.batch_size, n_linked))
        remaining = self.batch_size - len(selected_linked)
        selected_unlinked = (
            random.sample(list(unlinked_pool), min(remaining, n_unlinked))
            if remaining > 0 else []
        )

        selected_indices = selected_linked + selected_unlinked
        source_map: Dict[int, str] = {idx: "linked" for idx in selected_linked}
        source_map.update({idx: "unlinked" for idx in selected_unlinked})

        print(
            f"    [DC-SELECT E{epoch}|S{step_idx}] "
            f"changed_eids={changed_eids}  "
            f"linked_pool={n_linked}  unlinked_pool={n_unlinked}  "
            f"selected={len(selected_indices)} "
            f"(linked={len(selected_linked)}, unlinked={len(selected_unlinked)}) "
            f"— LOCKED for this step"
        )
        self._log(self.step_log,
            f"\n  [DC-SELECT E{epoch}|S{step_idx}] samples locked for this step\n"
            f"    changed_eids_at_selection={changed_eids}\n"
            f"    linked_pool_size={n_linked}  unlinked_pool_size={n_unlinked}\n"
            f"    selected={len(selected_indices)} indices={selected_indices}\n"
            f"    source_map={source_map}"
        )

        return selected_indices, source_map

    # ---------- run DC check with locked sample set ----------
    def _run_dc_check(
        self,
        selected_indices: List[int],
        source_map: Dict[int, str],
        working_rem: ReformulationExperienceMemory,
        env,
        step_idx: int,
        epoch: int,
        inner_iter: int,
    ) -> Tuple[bool, List[Dict[str, Any]], List[int], List[ReformulatorOutput]]:
        """
        Evaluate the locked DC sample set under working_rem.
        Returns: (all_passed, dc_failure_details, dc_ems, dc_refm_outs)
        """
        if not selected_indices:
            self._log(self.step_log,
                f"  [DC E{epoch}|S{step_idx}|I{inner_iter}] "
                f"No samples — DC skipped (auto-pass).")
            print(f"    [DC E{epoch}|S{step_idx}|I{inner_iter}] No samples — auto-pass.")
            return True, [], [], []

        print(
            f"    [DC E{epoch}|S{step_idx}|I{inner_iter}] "
            f"Re-evaluating {len(selected_indices)} locked samples: {selected_indices}"
        )
        self._log(self.step_log,
            f"\n  [DC E{epoch}|S{step_idx}|I{inner_iter}] "
            f"Re-evaluating {len(selected_indices)} locked samples: {selected_indices}"
        )

        dc_batch = [self.val_samples[i] for i in selected_indices]
        old_mode = self._token_mode
        self._token_mode = "dc"
        dc_results = self._evaluate_batch(dc_batch, working_rem, env)
        self._token_mode = old_mode

        dc_ems: List[int] = [r[0] for r in dc_results]
        dc_refm_outs: List[ReformulatorOutput] = [r[3] for r in dc_results]
        all_passed = all(em == 1 for em in dc_ems)

        dc_failure_details: List[Dict[str, Any]] = []
        for i, (val_idx, em, r) in enumerate(zip(selected_indices, dc_ems, dc_results)):
            src = source_map.get(val_idx, "unlinked")
            verdict = "PASS" if em == 1 else "FAIL"
            unc_type = self.val_samples[val_idx].get("uncertainty_type")
            matched = r[3].matched_experience_ids
            self._log(self.step_log,
                f"    DC[{i}] val_idx={val_idx} source={src} em={em} [{verdict}] "
                f"matched_ids={matched}")
            print(
                f"      [DC] em={em} [{verdict}] | uncertainty_type={unc_type} | "
                f"matched_ids={matched} | val_idx={val_idx}"
            )
            if em == 0:
                dc_failure_details.append({
                    "source": src,
                    "val_idx": val_idx,
                    "question": str(self.val_samples[val_idx].get("description", "")),
                    "reformulator_raw_output": r[3].raw_text,
                    "environment_feedback": r[1],
                    "em": em,
                })

        status_str = "PASSED" if all_passed else f"FAILED ({len(dc_failure_details)} regressions)"
        self._log(self.step_log, f"    [DC] Result: {status_str}")
        print(f"    [DC] Result: {status_str}")

        return all_passed, dc_failure_details, dc_ems, dc_refm_outs

    # ---------- update exp_val_correct_map after successful DC commit ----------
    def _commit_dc_map_update(
        self,
        changed_eids: List[int],
        selected_indices: List[int],
        dc_ems: List[int],
        dc_refm_outs: List[ReformulatorOutput],
    ) -> None:
        """
        1. Clear old mappings for all changed experiences (they've been modified).
        2. For each DC sample that was correct, add it to the mapping of the
           changed experience(s) it matched. If it matched none, attribute to all
           changed_eids (it was tested in their context).
        """
        for eid in changed_eids:
            old_size = len(self.exp_val_correct_map.pop(eid, set()))
            self._log(self.step_log,
                f"    [DC Map] Cleared exp_val_correct_map[{eid}] (had {old_size} entries)")

        for val_idx, em, refm_out in zip(selected_indices, dc_ems, dc_refm_outs):
            self.val_sample_results[val_idx] = em
            if em == 1:
                matched = [eid for eid in refm_out.matched_experience_ids if eid in changed_eids]
                if not matched:
                    matched = changed_eids
                for eid in matched:
                    self.exp_val_correct_map.setdefault(eid, set()).add(val_idx)

        self._log(self.step_log,
            f"    [DC Map] Updated for eids={changed_eids}. "
            f"New sizes: { {eid: len(self.exp_val_correct_map.get(eid, set())) for eid in changed_eids} }")

    # ---------- process one step (single training sample) ----------
    def _process_step(
        self,
        sample: Dict[str, Any],
        step_idx: int,
        epoch: int,
        env,
    ) -> Tuple[int, int, int, int, bool]:
        """
        Runs one training sample through the DC-gated reflection loop.
        Returns (1, initial_correct, final_correct, code_errors, memory_changed).
        """
        self._log(self.step_log, "\n" + "=" * 60)
        self._log(self.step_log, f"[Epoch {epoch} | Step {step_idx}/{STEPS_PER_EPOCH}]")
        self._log(self.step_log,
            f"  sample uncertainty_type={sample.get('uncertainty_type')}")

        # ── Initial evaluation ────────────────────────────────────────
        print(f"\n  [Step {step_idx}/{STEPS_PER_EPOCH}] Evaluating training sample...")
        em, feedback, error_type, refm_out, _ = self._run_pipeline_single(sample, self.rem, env)
        code_errors = 1 if error_type in ("code_error", "parse_error") else 0

        self._log(self.step_log,
            f"  Initial eval: em={em}, error_type={error_type}, "
            f"matched_ids={refm_out.matched_experience_ids}")
        self._log(self.step_log, f"  [Feedback]\n{feedback}")
        print(
            f"    [Train] em={em} | uncertainty_type={sample.get('uncertainty_type')} | "
            f"matched_ids={refm_out.matched_experience_ids}"
        )

        if em == 1:
            for eid in refm_out.matched_experience_ids:
                self.rem.record_usage_success(eid)
            self._log(self.step_log, "  → Correct. Skip reflection.")
            print(f"  [Step {step_idx}] Correct — skip.")
            return 1, 1, 1, code_errors, False

        for eid in refm_out.matched_experience_ids:
            self.rem.record_usage_failure(eid)

        # ── Reflection + DC inner loop ────────────────────────────────
        working_rem = ReformulationExperienceMemory.loads(self.rem.dumps())
        prev_train_trace: Optional[str] = None
        prev_dc_failures: Optional[List[Dict[str, Any]]] = None
        committed = False
        cur_em, cur_fb, cur_refm = em, feedback, refm_out

        # Cumulative tracking of what working_rem differs from self.rem
        cumul_changed: Set[int] = set()
        cumul_deleted: Set[int] = set()

        # DC sample set — selected once on first DC trigger and locked for the entire step
        locked_dc_indices: Optional[List[int]] = None
        locked_dc_source_map: Optional[Dict[int, str]] = None

        for inner_iter in range(1, MAX_INNER_ITERS + 1):
            self.total_inner_iterations += 1
            if inner_iter == MAX_INNER_ITERS:
                self.total_reach_max_inner_iters += 1

            print(f"\n    [Inner Iter {inner_iter}/{MAX_INNER_ITERS}]")
            self._log(self.step_log, f"\n  [Inner Iteration {inner_iter}/{MAX_INNER_ITERS}]")

            # Reflect
            refl_out: ReflectorOutput = self.reflector.generate(
                questions=[str(sample["description"])],
                rem=working_rem,
                reformulator_outputs=[cur_refm],
                environment_feedbacks=[cur_fb],
                ems=[cur_em],
                previous_reflector_trace=prev_train_trace,
                double_check_failures=prev_dc_failures,
            )
            self.total_refl_prompt_tokens += refl_out.prompt_tokens or 0
            self.total_refl_completion_tokens += refl_out.completion_tokens or 0
            self.total_refl_latency += refl_out.latency or 0.0

            proposed_ops = refl_out.final_answer if isinstance(refl_out.final_answer, list) else []
            ops = []
            for op in proposed_ops:
                if not isinstance(op, dict) or op.get("operator") not in ("add", "update", "delete"):
                    continue
                if op["operator"] == "add":
                    if _extract_content_from_op(op) is not None:
                        # The operation applicator assigns the next available ID.
                        ops.append(dict(op))
                    continue
                try:
                    experience_id = int(op["experience_id"])
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                if experience_id < 1:
                    continue
                if op["operator"] in ("add", "update") and _extract_content_from_op(op) is None:
                    continue
                ops.append({**op, "experience_id": experience_id})

            self._log(self.step_log, "\n  [Reflector Output]")
            self._log(self.step_log, "  --- Reasoning ---")
            self._log(self.step_log, refl_out.reasoning or "(no reasoning)")
            self._log(self.step_log, "\n  --- Operations ---")
            self._log(self.step_log,
                json.dumps(ops, indent=2, ensure_ascii=False) if ops else "(no operations)")

            for op in ops:
                if not isinstance(op, dict):
                    print(f"        op (invalid, skipped): {op!r}")
                    self._log(self.step_log, f"    op (invalid, skipped): {op!r}")
                    continue
                print(f"        op: {op.get('operator')} id={op.get('experience_id')}")
                self._log(self.step_log,
                    f"    op: {op.get('operator')} id={op.get('experience_id')}")

            self._apply_reflector_ops(ops, working_rem)

            # Update cumulative tracking (deferred — only applied to exp_val_correct_map at commit)
            for op in ops:
                if not isinstance(op, dict):
                    continue
                eid_raw = op.get("experience_id")
                if eid_raw is None:
                    continue
                eid = int(eid_raw)
                operator = op.get("operator")
                if operator in ("add", "update"):
                    cumul_changed.add(eid)
                    cumul_deleted.discard(eid)
                elif operator == "delete":
                    cumul_deleted.add(eid)
                    cumul_changed.discard(eid)

            print(f"      REM size after reflection: {len(working_rem.list_experience_ids())}")
            self._log(self.step_log,
                f"  REM size after reflection: {len(working_rem.list_experience_ids())}")

            # Re-evaluate training sample on working_rem
            re_em, re_fb, re_et, re_refm, _ = self._run_pipeline_single(sample, working_rem, env)
            print(
                f"      [Re-eval Train] em={re_em} | uncertainty_type={sample.get('uncertainty_type')} | "
                f"matched_ids={re_refm.matched_experience_ids}"
            )
            self._log(self.step_log, f"  Re-eval training sample: em={re_em}")

            # Build train trace for next iteration (NO DC info mixed in here)
            prev_train_trace = json.dumps({
                "inner_iteration": inner_iter,
                "operations_applied": [
                    {"operator": op.get("operator"), "experience_id": op.get("experience_id")}
                    for op in ops if isinstance(op, dict)
                ],
                "training_sample_em_after_ops": re_em,
                "training_sample_feedback_snippet": re_fb[:300] if re_fb else "",
            }, indent=2, ensure_ascii=False)
            prev_dc_failures = None

            if re_em == 0:
                cur_em, cur_fb, cur_refm = re_em, re_fb, re_refm
                self._log(self.step_log,
                    "  Training sample still incorrect — continuing inner loop.")
                continue

            # Training sample correct → proceed to Double Check
            all_affected_eids = list(cumul_changed | cumul_deleted)

            if not all_affected_eids:
                # Pure no-op — memory is unchanged, do NOT commit
                self._log(self.step_log,
                    "  No operations applied and training sample now correct "
                    "— memory unchanged, skipping commit.")
                print(f"    [Step {step_idx}] No ops applied — memory unchanged, not committing.")
                break

            changed_eids = list(cumul_changed)

            # Lock DC samples on first trigger (covers both add/update and delete ops);
            # reuse same set on subsequent inner iters
            if locked_dc_indices is None:
                locked_dc_indices, locked_dc_source_map = self._select_dc_samples(
                    all_affected_eids, step_idx, epoch
                )
                self.total_dc_checks += 1

            dc_passed, dc_failures, iter_dc_ems, iter_dc_refm_outs = self._run_dc_check(
                locked_dc_indices, locked_dc_source_map,
                working_rem, env, step_idx, epoch, inner_iter,
            )

            if dc_passed:
                self.total_dc_passed += 1
                self.rem = working_rem
                for eid in cumul_deleted:
                    self.exp_val_correct_map.pop(eid, None)
                self._commit_dc_map_update(
                    changed_eids, locked_dc_indices, iter_dc_ems, iter_dc_refm_outs
                )
                committed = True
                self._log(self.step_log,
                    f"  DC PASSED — memory committed. "
                    f"changed_eids={changed_eids} deleted_eids={list(cumul_deleted)}")
                print(
                    f"    [Step {step_idx}] DC PASSED — committed. "
                    f"changed_eids={changed_eids}")
                break
            else:
                self.total_dc_failed += 1
                prev_dc_failures = dc_failures
                cur_em, cur_fb, cur_refm = re_em, re_fb, re_refm
                self._log(self.step_log,
                    f"  DC FAILED ({len(dc_failures)} regressions) — continuing inner loop.")
                print(
                    f"    [Step {step_idx}] DC FAILED ({len(dc_failures)} regressions) — "
                    f"continuing inner loop.")

        if not committed:
            self._log(self.step_log,
                f"  Inner loop exhausted ({MAX_INNER_ITERS} iters) without commit — "
                f"working_rem discarded.")
            print(f"    [Step {step_idx}] Exhausted inner iters — no commit.")

        final_correct = 1 if committed else 0
        return 1, 0, final_correct, code_errors, committed

    # ---------- init val map (before training) ----------
    def _init_val_map(self, env) -> float:
        """
        Run full val set once to initialize val_sample_results and exp_val_correct_map.
        Called only on fresh start (not on resume).
        Returns initial val accuracy.
        """
        if not self.val_samples:
            self._log(self.train_log, "[Init Val] No val samples — skipping.")
            return 0.0
        total = len(self.val_samples)
        print(f"\n[Init Val] Running val set ({total} samples) to initialize DC map...")
        self._log(self.train_log,
            f"[Init Val] Running val set ({total} samples) to initialize DC map...")
        init_acc = self._evaluate_val_set(env)
        print(
            f"[Init Val] Done. val_sample_results={len(self.val_sample_results)} entries, "
            f"exp_val_correct_map={len(self.exp_val_correct_map)} experience entries, "
            f"init_val_acc={init_acc:.3f}"
        )
        self._log(self.train_log,
            f"[Init Val] Done. exp_val_correct_map entries={len(self.exp_val_correct_map)}, "
            f"init_val_acc={init_acc:.3f}")
        return init_acc

    # ---------- evaluate full validation set ----------
    def _evaluate_val_set(self, env) -> float:
        """
        Evaluate current REM on the full validation set.
        Atomically rebuilds exp_val_correct_map and val_sample_results.
        Returns accuracy (0.0–1.0).
        """
        if not self.val_samples:
            return 0.0

        total = len(self.val_samples)
        val_rem = ReformulationExperienceMemory.loads(self.rem.dumps())
        old_mode = self._token_mode
        self._token_mode = "val"
        results: List[Optional[Any]] = [None] * total
        print(f"    [Val] Launching {total} threads...")
        with ThreadPoolExecutor(max_workers=total) as executor:
            futures = {
                executor.submit(self._run_pipeline_single, self.val_samples[i], val_rem, env): i
                for i in range(total)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        self._token_mode = old_mode

        correct = 0
        new_exp_val_map: Dict[int, Set[int]] = {}
        val_ems: List[int] = []
        for i, res in enumerate(results):
            em = res[0]
            refm_out = res[3]
            correct += em
            val_ems.append(em)
            self.val_sample_results[i] = em
            if em == 1:
                for eid in refm_out.matched_experience_ids:
                    new_exp_val_map.setdefault(eid, set()).add(i)

        self.exp_val_correct_map = new_exp_val_map

        # Format results into 4 blocks for clarity
        chunk_size = (total + 3) // 4  # ceiling division
        val_ems_str = " | ".join(
            ''.join(str(e) for e in val_ems[i*chunk_size:(i+1)*chunk_size])
            for i in range(4)
        )
        print(f"  [Val] Results: {val_ems_str}")
        self._log(self.train_log, f"  [Val] Results: {val_ems_str}")

        acc = correct / total if total > 0 else 0.0
        print(f"  [Val] {correct}/{total} correct, acc={acc:.3f}")
        self._log(self.train_log, f"  [Val] {correct}/{total} correct, val_acc={acc:.3f}")
        self._log(self.train_log,
            f"  [Val] exp_val_correct_map rebuilt: "
            f"{len(self.exp_val_correct_map)} experience entries, "
            f"correct val samples={correct}")
        return acc

    # ---------- generate one training sample ----------
    @staticmethod
    def _generate_sample() -> Dict[str, Any]:
        """Generate a single fresh training sample."""
        var_list = [2, 3, 4, 5]
        template_list = [
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ]
        chosen_var = random.choice(var_list)
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
            pool = batch_generate_json(
                [chosen_var], 1, template_list=template_list, mode="random"
            )
        sample = random.choice(pool)
        print(f"[Data] Sample generated: var={chosen_var}")
        return sample

    # ---------- serialize DC map for checkpoint ----------
    def _exp_val_map_to_json(self) -> Dict[str, List[int]]:
        return {str(k): sorted(v) for k, v in self.exp_val_correct_map.items()}

    # ---------- full training loop ----------
    def run(self, env) -> None:
        self.train_start_time = time.time()
        self._log(self.train_log,
            f"[Training Start] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.train_start_time))}")
        self._log(self.train_log,
            f"Hyperparams: EPOCHS={self.epochs}, STEPS_PER_EPOCH={STEPS_PER_EPOCH}, "
            f"dc_sample_size(batch_size)={self.batch_size}, MAX_INNER_ITERS={MAX_INNER_ITERS}")

        try:
            best_val_acc: float = self._resume_best_val_acc

            # Initialize DC map on fresh start
            if self.start_epoch == 1 and not self.val_sample_results:
                init_acc = self._init_val_map(env)
                best_val_acc = init_acc  # use real baseline, not 0
                self._log(self.train_log,
                    f"[Init] Initial val_acc={best_val_acc:.3f} (set as baseline for val-gate)")
                # Save initial best_rem (baseline before any training)
                init_best_path = self.save_dir / "best_rem.json"
                init_data = json.loads(self.rem.dumps())
                init_data["best_val_acc"] = best_val_acc
                init_best_path.write_text(
                    json.dumps(init_data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self._log(self.train_log,
                    f"[Init] Saved initial best_rem.json (pre-training baseline)")

            for epoch in range(self.start_epoch, self.epochs + 1):
                print(f"\n{'='*30}")
                print(f" Offline Epoch {epoch}/{self.epochs}")
                print(f"{'='*30}")
                self._log(self.train_log, f"\n[Epoch {epoch}]")

                pre_epoch_rem_snapshot = self.rem.dumps()
                pre_epoch_dc_map: Dict[int, Set[int]] = {
                    k: set(v) for k, v in self.exp_val_correct_map.items()
                }
                pre_epoch_vsr: Dict[int, int] = dict(self.val_sample_results)

                epoch_init_correct = 0
                epoch_final_correct = 0
                epoch_code_errors = 0
                epoch_total = 0
                epoch_mem_changed = False

                for step in range(1, STEPS_PER_EPOCH + 1):
                    print(f"\n  [Step {step}/{STEPS_PER_EPOCH}]")
                    self._log(self.train_log, f"  [Step {step}/{STEPS_PER_EPOCH}]")

                    sample = self._generate_sample()

                    n, init_c, final_c, code_err, mem_changed = self._process_step(
                        sample, step, epoch, env
                    )
                    epoch_total += n
                    epoch_init_correct += init_c
                    epoch_final_correct += final_c
                    epoch_code_errors += code_err
                    epoch_mem_changed |= mem_changed

                    print(
                        f"  [Step {step}] Done: init={init_c}/{n}, final={final_c}/{n}, "
                        f"REM size={len(self.rem.list_experience_ids())}"
                    )
                    self._log(self.train_log,
                        f"    step_init_correct={init_c}, step_final_correct={final_c}, "
                        f"rem_size={len(self.rem.list_experience_ids())}")

                # ── Epoch summary ─────────────────────────────────────
                acc_init = epoch_init_correct / epoch_total if epoch_total else 0.0
                acc_final = epoch_final_correct / epoch_total if epoch_total else 0.0
                print(f"\n[Epoch {epoch}] GT Acc (init)  = {epoch_init_correct}/{epoch_total} = {acc_init:.3f}")
                print(f"[Epoch {epoch}] GT Acc (final) = {epoch_final_correct}/{epoch_total} = {acc_final:.3f}")
                print(f"[Epoch {epoch}] Code errors = {epoch_code_errors}/{epoch_total}")
                print(f"[Epoch {epoch}] REM size = {len(self.rem.list_experience_ids())}")
                print(
                    f"[Epoch {epoch}] DC stats (cumulative): "
                    f"checks={self.total_dc_checks}, "
                    f"passed={self.total_dc_passed}, "
                    f"failed={self.total_dc_failed}"
                )

                self._log(self.train_log, f"  train_correct_init={epoch_init_correct}")
                self._log(self.train_log, f"  train_correct_final={epoch_final_correct}")
                self._log(self.train_log, f"  train_total={epoch_total}")
                self._log(self.train_log, f"  train_acc_init={acc_init:.3f}")
                self._log(self.train_log, f"  train_acc_final={acc_final:.3f}")
                self._log(self.train_log, f"  code_errors={epoch_code_errors}")
                self._log(self.train_log, f"  rem_size={len(self.rem.list_experience_ids())}")
                self._log(self.train_log,
                    f"  [DC] total_checks={self.total_dc_checks} "
                    f"passed={self.total_dc_passed} failed={self.total_dc_failed}")

                # ── Metrics log ───────────────────────────────────────
                self.metric_log.write(
                    f"{epoch},{acc_init:.3f},{acc_final:.3f},"
                    f"{epoch_init_correct},{epoch_final_correct},{epoch_total},"
                    f"{epoch_code_errors},{len(self.rem.list_experience_ids())},"
                    f"{self.total_refm_prompt_tokens},{self.total_refm_completion_tokens},{self.total_refm_latency:.2f},"
                    f"{self.total_coder_prompt_tokens},{self.total_coder_completion_tokens},{self.total_coder_latency:.2f},"
                    f"{self.total_refl_prompt_tokens},{self.total_refl_completion_tokens},{self.total_refl_latency:.2f},"
                    f"{self.train_fc_prompt_tokens},{self.train_fc_completion_tokens},{self.train_fc_latency:.2f},"
                    f"{self.val_fc_prompt_tokens},{self.val_fc_completion_tokens},{self.val_fc_latency:.2f},"
                    f"{self.dc_fc_prompt_tokens},{self.dc_fc_completion_tokens},{self.dc_fc_latency:.2f}\n"
                )
                self.metric_log.flush()

                self._log(self.train_log,
                    f"  [Inner Loop] total_inner_iterations={self.total_inner_iterations}, "
                    f"reach_max={self.total_reach_max_inner_iters}")
                self._log(self.train_log,
                    f"  [Reformulator] pt={self.total_refm_prompt_tokens} "
                    f"ct={self.total_refm_completion_tokens} lat={self.total_refm_latency:.2f}s")
                self._log(self.train_log,
                    f"  [Coder] pt={self.total_coder_prompt_tokens} "
                    f"ct={self.total_coder_completion_tokens} lat={self.total_coder_latency:.2f}s")
                self._log(self.train_log,
                    f"  [FC train] pt={self.train_fc_prompt_tokens} "
                    f"ct={self.train_fc_completion_tokens} lat={self.train_fc_latency:.2f}s")
                self._log(self.train_log,
                    f"  [FC val] pt={self.val_fc_prompt_tokens} "
                    f"ct={self.val_fc_completion_tokens} lat={self.val_fc_latency:.2f}s")
                self._log(self.train_log,
                    f"  [FC DC] pt={self.dc_fc_prompt_tokens} "
                    f"ct={self.dc_fc_completion_tokens} lat={self.dc_fc_latency:.2f}s")
                self._log(self.train_log,
                    f"  [Reflector] pt={self.total_refl_prompt_tokens} "
                    f"ct={self.total_refl_completion_tokens} lat={self.total_refl_latency:.2f}s")

                # ── Val-gated memory acceptance + checkpoint ──────────
                save_path = self.save_dir / f"trained_rem_epoch{epoch}.json"

                def _save_checkpoint(rem_to_save, bva, _path=save_path):
                    data = json.loads(rem_to_save.dumps())
                    data["best_val_acc"] = bva
                    data["exp_val_correct_map"] = self._exp_val_map_to_json()
                    data["val_sample_results"] = {
                        str(k): v for k, v in self.val_sample_results.items()
                    }
                    _path.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                    )

                def _save_best_rem(rem_to_save, bva):
                    """Save inference-ready memory (no DC tracking state)."""
                    data = json.loads(rem_to_save.dumps())
                    data["best_val_acc"] = bva
                    best_path = self.save_dir / "best_rem.json"
                    best_path.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    print(f"[Best REM] Saved to {best_path} (val_acc={bva:.3f})")
                    self._log(self.train_log,
                        f"  [Best REM] Saved to {best_path} (val_acc={bva:.3f})")

                if not epoch_mem_changed:
                    print(
                        f"\n[Epoch {epoch}] Memory unchanged — "
                        f"skipping validation (best_val_acc={best_val_acc:.3f})."
                    )
                    self._log(self.train_log,
                        f"  [Val-Gate] Memory unchanged — validation skipped, "
                        f"best_val_acc stays {best_val_acc:.3f}")
                else:
                    print(f"\n[Epoch {epoch}] Running full validation set evaluation...")
                    self._log(self.train_log,
                        f"  [Val-Gate] Evaluating on validation set...")
                    val_acc = self._evaluate_val_set(env)
                    if val_acc >= best_val_acc:
                        # Accept on improvement OR tie (equal performance keeps latest memory)
                        print(
                            f"[Epoch {epoch}] Val acc {best_val_acc:.3f} → {val_acc:.3f}. "
                            f"Memory accepted."
                        )
                        self._log(self.train_log,
                            f"  [Val-Gate] ACCEPTED: "
                            f"val_acc {best_val_acc:.3f} → {val_acc:.3f}")
                        best_val_acc = val_acc
                        # exp_val_correct_map already refreshed by _evaluate_val_set
                        _save_best_rem(self.rem, best_val_acc)
                    else:
                        print(
                            f"[Epoch {epoch}] Val acc did NOT improve: "
                            f"{val_acc:.3f} < {best_val_acc:.3f}. Rolling back."
                        )
                        self._log(self.train_log,
                            f"  [Val-Gate] REJECTED: "
                            f"val_acc {val_acc:.3f} < best {best_val_acc:.3f}. "
                            f"Rolling back.")
                        self.rem = ReformulationExperienceMemory.loads(
                            pre_epoch_rem_snapshot)
                        self.exp_val_correct_map = pre_epoch_dc_map
                        self.val_sample_results = pre_epoch_vsr

                if best_val_acc == 1.0:
                    print(f"\n[Epoch {epoch}] Early stopping: val acc reached 1.0.")
                    self._log(self.train_log,
                        f"  [Early Stop] Val acc={best_val_acc:.3f} — stopping.")
                    _save_checkpoint(self.rem, best_val_acc)
                    _save_best_rem(self.rem, best_val_acc)
                    print(f"[Checkpoint] REM saved to {save_path}")
                    self._log(self.train_log, f"  [Checkpoint] REM saved to {save_path}")
                    return

                _save_checkpoint(self.rem, best_val_acc)

                print(f"[Checkpoint] REM saved to {save_path}")
                self._log(self.train_log, f"  [Checkpoint] REM saved to {save_path}")

        except LLMNetworkError as e:
            print(f"\n[INTERRUPT] Training interrupted due to persistent network error: {e}")
            self._log(self.train_log,
                f"[INTERRUPT] Training stopped due to network error: {e}")
            raise

        finally:
            train_end_time = time.time()
            duration = train_end_time - self.train_start_time
            self._log(self.train_log,
                f"[Training End] "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_end_time))}")
            self._log(self.train_log, f"[Training Duration] {duration:.2f} seconds")
            self.metric_log.write(f"# wall_clock_seconds={duration:.2f}\n")
            self.metric_log.flush()
            for handle in (self.train_log, self.step_log, self.failed_log, self.metric_log):
                handle.close()
