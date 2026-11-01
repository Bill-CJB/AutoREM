"""Offline memory adaptation for AutoREM."""

from __future__ import annotations

from pathlib import Path


def train(args) -> None:
    from .adaptation import OfflineAdapter
    from .llm import LLMClient
    from .memory import ReformulationExperienceMemory
    from .robust_env import RobustTaskEnvironment
    from .roles import Coder, Reflector, Reformulator

    base_client = LLMClient.from_env()
    coder_client = LLMClient.from_env(coder=True)
    adapter = OfflineAdapter(
        rem=ReformulationExperienceMemory(),
        reformulator=Reformulator(base_client),
        coder=Coder(coder_client),
        reflector=Reflector(base_client),
        save_dir=str(Path(args.output)),
        val_data=str(args.validation),
        epochs=10,
        batch_size=4,
        resume_from=str(args.resume) if args.resume else None,
    )
    try:
        adapter.run(RobustTaskEnvironment())
    finally:
        for name in ("train_log", "step_log", "failed_log", "metric_log"):
            getattr(adapter, name).close()
    print(f"Checkpoints: {adapter.save_dir}")
