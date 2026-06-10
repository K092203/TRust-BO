from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Trial:
    trial_id: int
    parameters: dict[str, Any]           # 元の値(永遠の正規形)
    objective_values: list[float] | None = None
    constraint_values: list[float] | None = None
    status: str = "complete"             # complete | failed | infeasible
    failure_reason: str | None = None
    metadata: dict | None = None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class HistoryStore:
    def __init__(self) -> None:
        self._trials: list[Trial] = []
        self._next_id: int = 0

    def add(self, trial: Trial) -> None:
        self._trials.append(trial)
        self._next_id = max(self._next_id, trial.trial_id + 1)

    def next_id(self) -> int:
        return self._next_id

    def all_trials(self) -> list[Trial]:
        return list(self._trials)

    def complete_trials(self) -> list[Trial]:
        return [t for t in self._trials if t.status == "complete"]

    def feasible_trials(self) -> list[Trial]:
        return [
            t for t in self._trials
            if t.status == "complete"
            and (t.constraint_values is None or all(v <= 0 for v in t.constraint_values))
        ]

    # --- 永続化 ---

    def to_jsonl(self) -> str:
        lines = []
        for t in self._trials:
            lines.append(json.dumps({
                "trial_id": t.trial_id,
                "parameters": t.parameters,
                "objective_values": t.objective_values,
                "constraint_values": t.constraint_values,
                "status": t.status,
                "failure_reason": t.failure_reason,
                "metadata": t.metadata,
                "started_at": t.started_at,
                "completed_at": t.completed_at,
            }))
        return "\n".join(lines)

    @classmethod
    def from_jsonl(cls, text: str) -> HistoryStore:
        store = cls()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            trial = Trial(
                trial_id=d["trial_id"],
                parameters=d["parameters"],
                objective_values=d.get("objective_values"),
                constraint_values=d.get("constraint_values"),
                status=d.get("status", "complete"),
                failure_reason=d.get("failure_reason"),
                metadata=d.get("metadata"),
                started_at=d.get("started_at", 0.0),
                completed_at=d.get("completed_at"),
            )
            store.add(trial)
        return store

    def save_jsonl(self, path: str | Path) -> None:
        Path(path).write_text(self.to_jsonl(), encoding="utf-8")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> HistoryStore:
        return cls.from_jsonl(Path(path).read_text(encoding="utf-8"))
