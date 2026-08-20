from __future__ import annotations

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any
from datetime import datetime, timezone
from pathlib import Path
import json


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$")
    instruction: str = Field(min_length=5)
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=5)
    tasks: list[Task] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_dag(self) -> "Plan":
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Task ids must be unique")
        known = set(ids)
        for task in self.tasks:
            missing = set(task.depends_on) - known
            if missing:
                raise ValueError(f"{task.id} has unknown dependencies: {sorted(missing)}")
            if task.id in task.depends_on:
                raise ValueError(f"{task.id} cannot depend on itself")
        if not nx.is_directed_acyclic_graph(self.graph):
            cycle = nx.find_cycle(self.graph)
            blocked = sorted({node for edge in cycle for node in edge[:2]})
            raise ValueError(f"Cycle detected; blocked tasks: {blocked}")
        return self

    @property
    def graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(task.id for task in self.tasks)
        graph.add_edges_from(
            (dependency, task.id)
            for task in self.tasks
            for dependency in task.depends_on
        )
        return graph

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self.graph))

    def execution_batches(self) -> list[list[str]]:
        return [sorted(generation) for generation in nx.topological_generations(self.graph)]

    def task(self, task_id: str) -> Task:
        return next(task for task in self.tasks if task.id == task_id)

    def terminal_tasks(self) -> list[str]:
        return [node for node, degree in self.graph.out_degree if degree == 0]


class Thought(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class EnvironmentFeedback(BaseModel):
    """A grounded signal produced outside the language model."""

    success: bool
    score: float = Field(ge=0.0, le=1.0)
    details: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class TraceEvent(BaseModel):
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)


class BenchmarkObservation(BaseModel):
    """One measured method/case result stored inside the normal run trace."""

    benchmark: str
    case_id: str
    method: str
    success: bool
    llm_calls: int | None = None
    llm_calls_label: str | None = None
    tokens: int | None = None
    latency_seconds: float | None = None
    estimated_cost: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)


class RunTrace(BaseModel):
    """Canonical evidence format for every run; benchmark data extends this same trace."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mode: str
    model: str | None = None
    goal: str | None = None
    plans: list[dict[str, Any]] = Field(default_factory=list)
    node_outputs: list[dict[str, Any]] = Field(default_factory=list)
    critic_feedback: list[dict[str, Any]] = Field(default_factory=list)
    episodic_memories: list[dict[str, Any]] = Field(default_factory=list)
    mcts_visits: list[dict[str, Any]] = Field(default_factory=list)
    branch_reflections: list[dict[str, Any]] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)
    benchmark_observations: list[BenchmarkObservation] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)

    def add_event(self, kind: str, **payload: Any) -> None:
        self.events.append(TraceEvent(kind=kind, payload=payload))

    def add_benchmark(self, observation: BenchmarkObservation) -> None:
        self.benchmark_observations.append(observation)

    @classmethod
    def from_json(cls, path: Path) -> "RunTrace":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def artifact_dir(root: Path) -> Path:
    path = root / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_trace(trace: RunTrace, root: Path) -> Path:
    """Write the canonical run trace. Normal and benchmark runs use this one path."""
    path = artifact_dir(root) / f"run-{trace.run_id}.json"
    path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_traces(root: Path) -> list[RunTrace]:
    directory = artifact_dir(root)
    return [RunTrace.from_json(path) for path in sorted(directory.glob("run-*.json"))]


def aggregate_benchmarks(traces: list[RunTrace]) -> dict[str, list[dict[str, Any]]]:
    """Aggregate only observations found in artifacts; no benchmark numbers are stored here."""
    grouped: dict[tuple[str, str], list[BenchmarkObservation]] = {}
    for trace in traces:
        for item in trace.benchmark_observations:
            grouped.setdefault((item.benchmark, item.method), []).append(item)

    reports: dict[str, list[dict[str, Any]]] = {}
    for (benchmark, method), observations in grouped.items():
        success = sum(item.success for item in observations)
        numeric = [item for item in observations if item.llm_calls is not None]
        row = {
            "method": method,
            "success": f"{success}/{len(observations)}",
            "success_rate": success / len(observations) if observations else 0.0,
            "avg_llm_calls": (
                sum(item.llm_calls for item in numeric) / len(numeric) if numeric else None
            ),
            "avg_tokens": (
                sum(item.tokens for item in numeric if item.tokens is not None)
                / len([item for item in numeric if item.tokens is not None])
                if any(item.tokens is not None for item in numeric) else None
            ),
            "avg_latency": (
                sum(item.latency_seconds for item in numeric if item.latency_seconds is not None)
                / len([item for item in numeric if item.latency_seconds is not None])
                if any(item.latency_seconds is not None for item in numeric) else None
            ),
            "avg_cost": (
                sum(item.estimated_cost for item in numeric if item.estimated_cost is not None)
                / len([item for item in numeric if item.estimated_cost is not None])
                if any(item.estimated_cost is not None for item in numeric) else None
            ),
            "llm_calls_label": (observations[0].llm_calls_label if observations and all(item.llm_calls_label == observations[0].llm_calls_label for item in observations) else None),
            "cases": len(observations),
        }
        reports.setdefault(benchmark, []).append(row)
    return reports


def markdown_report(reports: dict[str, list[dict[str, Any]]]) -> str:
    """Render comparison tables from trace observations."""
    chunks: list[str] = []
    labels = {
        "tuesday_reshuffle": "Top-level decomposition: reshuffling Tuesday's board (20 real cases)",
        "planning_subtasks": "Planning the rank-by-urgency and propose-the-reshuffle sub-tasks (15 cases each)",
    }
    for benchmark, rows in reports.items():
        chunks.append(f"## {labels.get(benchmark, benchmark)}")
        chunks.append("| Method | Task success | Avg. LLM calls | Avg. tokens | Avg. latency | Est. cost/run |")
        chunks.append("|---|---:|---:|---:|---:|---:|")
        for row in rows:
            calls = row["avg_llm_calls"]
            calls_label = row.get("llm_calls_label")
            tokens = row["avg_tokens"]
            latency = row["avg_latency"]
            cost = row["avg_cost"]
            chunks.append(
                f"| {row['method']} | {row['success']} | "
                f"{calls_label or f'{calls:.1f}'} | {tokens:,.0f} | {latency:.2f}s | ${cost:.2f} |"
                if calls is not None and tokens is not None and latency is not None and cost is not None
                else f"| {row['method']} | {row['success']} | - | - | - | - |"
            )
        chunks.append("")
    return "\n".join(chunks).strip()
