"""
Reformulation Experience Memory (REM)

Memory structure:
- Each experience has a single unified `content` field (plain text) that includes:
- Each experience applies to a minimal independent Formulation Unit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional


# =====================
# Experience definition
# =====================

@dataclass
class ReformulationExperience:
    experience_id: int
    content: str  # unified text with sections (a)–(d)
    usage_success_count: int = 0
    usage_failure_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ===============================
# Reformulation Experience Memory
# ===============================

class ReformulationExperienceMemory:
    def __init__(self) -> None:
        self._experiences: Dict[int, ReformulationExperience] = {}
        self._next_id: int = 1

    # ---- serialization ----
    def to_dict(self) -> Dict[str, object]:
        return {
            "experiences": {str(k): asdict(v) for k, v in self._experiences.items()},
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ReformulationExperienceMemory":
        inst = cls()
        exps = payload.get("experiences", {})
        if isinstance(exps, dict):
            for k, v in exps.items():
                if not isinstance(v, dict):
                    continue
                try:
                    eid = int(v.get("experience_id", k))
                except Exception:
                    continue

                inst._experiences[eid] = ReformulationExperience(
                    experience_id=eid,
                    content=str(v.get("content", "")),
                    usage_success_count=int(v.get("usage_success_count", 0)),
                    usage_failure_count=int(v.get("usage_failure_count", 0)),
                    created_at=v.get("created_at"),
                    updated_at=v.get("updated_at"),
                )
        if inst._experiences:
            inst._next_id = max(inst._experiences.keys()) + 1
        else:
            inst._next_id = 1
        return inst

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def loads(cls, data: str) -> "ReformulationExperienceMemory":
        return cls.from_dict(json.loads(data))

    # ---- prompt rendering ----
    def render_experiences(self, ids: Iterable[int]) -> str:
        """Render selected experiences (full content) for the Reformulator."""
        lines = []
        for eid in ids or []:
            eid = int(eid)
            if eid not in self._experiences:
                continue
            e = self._experiences[eid]
            lines.extend([
                f"## EXPERIENCE exp-{e.experience_id}",
                "",
                e.content,
                "",
                "----",
                "",
            ])
        return "\n".join(lines).strip()

    def render_catalog(self, *, show_scores: bool = True) -> str:
        """Render catalog (all experiences with content and optional quality scores)."""
        if not self._experiences:
            return ""

        lines = []
        for eid in sorted(self._experiences):
            e = self._experiences[eid]
            lines.append(f"- experience_id: {e.experience_id}")
            if show_scores:
                total = e.usage_success_count + e.usage_failure_count
                rate = (e.usage_success_count / total * 100) if total > 0 else 0.0
                if total > 0:
                    quality_tag = (
                        f"[Usage: {e.usage_success_count} success / {total} attempts, "
                        f"success_rate={rate:.0f}%]"
                    )
                else:
                    quality_tag = "[Usage: no usage yet]"
                lines.append(f"  {quality_tag}")
            lines.append("  content:")
            lines.append(e.content)
            lines.append("")
        return "\n".join(lines).strip()

    # ---- core CRUD ----
    def peek_next_id(self) -> int:
        """Return the next ID that will be auto-assigned without consuming it."""
        return self._next_id

    def _new_experience_id(self) -> int:
        eid = self._next_id
        self._next_id += 1
        return eid

    def add_experience(
        self,
        *,
        content: str,
        experience_id: Optional[int] = None,
    ) -> int:
        if experience_id is None:
            experience_id = self._new_experience_id()
        else:
            experience_id = int(experience_id)
            self._next_id = max(self._next_id, experience_id + 1)

        self._experiences[experience_id] = ReformulationExperience(
            experience_id=experience_id,
            content=content,
        )
        return experience_id

    def update_content(self, experience_id: int, content: str) -> None:
        """Replace the unified content of an existing experience and reset usage counters."""
        e = self._experiences.get(int(experience_id))
        if e is None:
            return
        e.content = content
        e.updated_at = datetime.now(timezone.utc).isoformat()
        e.usage_success_count = 0
        e.usage_failure_count = 0

    # ---- usage stats ----
    def set_usage_counts(self, experience_id: int, success: int, failure: int) -> None:
        """Directly set the usage counts (used to initialize after add/update)."""
        e = self._experiences.get(int(experience_id))
        if e is not None:
            e.usage_success_count = success
            e.usage_failure_count = failure

    def record_usage_success(self, experience_id: int) -> None:
        """Record that this experience was used and produced a correct final answer."""
        e = self._experiences.get(int(experience_id))
        if e is not None:
            e.usage_success_count += 1

    def record_usage_failure(self, experience_id: int) -> None:
        """Record that this experience was used and produced a wrong final answer."""
        e = self._experiences.get(int(experience_id))
        if e is not None:
            e.usage_failure_count += 1

    def get_score(self, experience_id: int) -> Dict[str, object]:
        e = self._experiences.get(int(experience_id))
        if e is None:
            return {"success": 0, "failure": 0, "total": 0, "success_rate": 0.0}
        total = e.usage_success_count + e.usage_failure_count
        rate = (e.usage_success_count / total) if total > 0 else 0.0
        return {
            "success": e.usage_success_count,
            "failure": e.usage_failure_count,
            "total": total,
            "success_rate": rate,
        }

    def remove_experience(self, experience_id: int) -> None:
        self._experiences.pop(int(experience_id), None)
        if self._experiences:
            self._next_id = max(self._experiences.keys()) + 1
        else:
            self._next_id = 1

    # ---- accessors ----
    def get_experience(self, experience_id: int) -> Optional[ReformulationExperience]:
        return self._experiences.get(int(experience_id))

    def list_experience_ids(self) -> list[int]:
        return sorted(self._experiences.keys())

    def all(self) -> Dict[int, ReformulationExperience]:
        return dict(self._experiences)
