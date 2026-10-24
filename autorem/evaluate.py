"""Evaluate AutoREM with a frozen reformulation memory."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .code_executor import execute_gurobi_code
from .llm import LLMClient
from .memory import ReformulationExperienceMemory
from .roles import Coder, Reformulator


def load_samples(path: Path) -> list[dict]:
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError("Dataset must be a nonempty JSON list.")
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict) or not isinstance(sample.get("description"), str):
            raise ValueError(f"Sample {i} must contain a description string.")
        value = sample.get("optimal_value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Sample {i} must contain a finite numeric optimal_value.")
    return samples


def evaluate_sample(sample: dict, *, memory, reformulator, coder, formulator_client=None) -> dict:
    start = time.perf_counter()
    question = sample["description"]
    formulation = None
    if formulator_client is not None:
        from .application import formulate
        formulation = formulate(formulator_client, question)
        question = formulation["final_answer"]
    reformulation = reformulator.generate(rem=memory, question=question)
    status = "truncated" if reformulation.is_truncated else "json_failed" if reformulation.is_json_failed else "ok"
    code = coder.generate(latex_formulation=reformulation.final_answer, formulation_status=status)
    objective, solution, error_type, error_message = execute_gurobi_code(code.final_answer)
    correct = error_type is None and objective is not None and abs(objective - sample["optimal_value"]) < 1e-4
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for stage in (formulation, asdict(reformulation), asdict(code)):
        if stage:
            for name in usage:
                usage[name] += int(stage.get(name) or 0)
    return {
        "id": sample.get("id"), "correct": bool(correct),
        "optimal_value": sample["optimal_value"], "predicted_value": objective,
        "solution": solution, "error_type": error_type, "error_message": error_message,
        "formulation": formulation, "reformulation": asdict(reformulation),
        "coding": asdict(code), "usage": usage,
        "wall_time_seconds": time.perf_counter() - start,
    }


def evaluate(args) -> None:
    dataset = Path(args.data) if args.data else Path("datasets") / (args.dataset + ".json")
    samples = load_samples(dataset)
    if args.limit:
        samples = samples[:args.limit]
    memory = ReformulationExperienceMemory.loads(Path(args.memory).read_text(encoding="utf-8"))
    if not memory.list_experience_ids():
        raise ValueError("The evaluation memory must contain AutoREM experiences.")
    base = LLMClient.from_env(task="application" if args.dataset == "application" else "reformulation")
    coder = Coder(LLMClient.from_env(coder=True))
    if args.dataset == "application":
        from .application import ApplicationReformulator
        reformulator = ApplicationReformulator(base)
        formulator = base
    else:
        reformulator = Reformulator(base)
        formulator = None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = [None] * len(samples)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(evaluate_sample, sample, memory=memory,
                               reformulator=reformulator, coder=coder,
                               formulator_client=formulator): i for i, sample in enumerate(samples)}
        for future in as_completed(pending):
            i = pending[future]
            # Infrastructure/API errors abort evaluation instead of becoming model errors.
            results[i] = future.result()
            (output / f"sample_{i + 1:04d}.json").write_text(
                json.dumps(results[i], indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Completed {sum(r is not None for r in results)}/{len(samples)}", flush=True)
    metrics = {
        "dataset": dataset.name, "num_samples": len(samples),
        "num_correct": sum(r["correct"] for r in results),
        "accuracy": sum(r["correct"] for r in results) / len(samples),
        "objective_tolerance": 1e-4, "comparison": "strict_absolute_error",
        "base_model": base.model, "coder_model": coder.client.model,
        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in results),
        "completion_tokens": sum(r["usage"]["completion_tokens"] for r in results),
        "wall_time_seconds": time.perf_counter() - start,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
