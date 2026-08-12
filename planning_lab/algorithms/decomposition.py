from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from ..models import Plan


PLANNER_SYSTEM = """You are a careful task-decomposition planner.
Produce a small executable DAG, not a prose checklist. Every task must make a concrete
contribution to the goal. Independent research or analysis tasks should be parallel.
The plan must end with exactly one synthesis task depending on every necessary branch."""


class PlannedTask(BaseModel):
    """Wire schema; richer semantic constraints are applied by the Task domain model."""

    model_config = ConfigDict(extra="forbid")

    id: str
    instruction: str
    depends_on: list[str]


class GeneratedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: list[PlannedTask]


def decompose_goal(goal: str, llm: BaseChatModel) -> Plan:
    generated = llm.with_structured_output(
        GeneratedPlan,
        method="json_schema",
    ).invoke([
        ("system", PLANNER_SYSTEM),
        ("human", f"""Decompose this goal into 3-6 tasks: {goal!r}
Use short task ids such as t1. Dependencies may refer only to tasks in the plan.
Preserve the supplied goal exactly in the plan's goal field."""),
    ], temperature=0.1)
    # The caller's goal remains authoritative even if the model paraphrases it.
    payload = generated.model_dump()
    payload["goal"] = goal
    return Plan.model_validate(payload)


def _run_node(plan: Plan, task_id: str, outputs: dict[str, str], llm: BaseChatModel) -> str:
    task = plan.task(task_id)
    context = "\n\n".join(
        f"OUTPUT FROM {dependency}:\n{outputs[dependency]}"
        for dependency in task.depends_on
    ) or "No prerequisite outputs."
    prompt = f"""Overall goal: {plan.goal}
        Current task: {task.instruction}
        Prerequisite outputs:
        {context}
        Complete only the current task. Be concrete and concise. Do not invent sources."""
    response = llm.invoke([
        ("system", "You execute one node in a validated task DAG."),
        ("human", prompt),
    ], temperature=0.2)
    content = response.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The chat model returned an empty or unsupported response")
    return content.strip()


def execute_plan(
    plan: Plan,
    llm: BaseChatModel,
    max_workers: int = 4,
    retries: int = 1,
) -> dict[str, str]:
    if retries < 0:
        raise ValueError("retries must be non-negative")
    outputs: dict[str, str] = {}
    for batch in plan.execution_batches():
        pending = set(batch)
        attempt = 0
        while pending and attempt <= retries:
            remaining = list(pending)
            failures: dict[str, Exception] = {}
            if max_workers > 1 and len(remaining) > 1:
                with ThreadPoolExecutor(max_workers=min(max_workers, len(remaining))) as pool:
                    futures = {
                        pool.submit(_run_node, plan, task_id, outputs, llm): task_id
                        for task_id in remaining
                    }
                    for future in as_completed(futures):
                        task_id = futures[future]
                        try:
                            outputs[task_id] = future.result()
                        except Exception as exc:
                            failures[task_id] = exc
            else:
                for task_id in remaining:
                    try:
                        outputs[task_id] = _run_node(plan, task_id, outputs, llm)
                    except Exception as exc:
                        failures[task_id] = exc
            pending = set(failures)
            attempt += 1
        if pending:
            detail = "; ".join(
                f"{task_id}: {failures[task_id]}" for task_id in sorted(pending)
            )
            raise RuntimeError(f"Task execution failed after {retries} retries: {detail}")
    return outputs


def final_output(plan: Plan, outputs: dict[str, str]) -> str:
    terminals = plan.terminal_tasks()
    if len(terminals) != 1:
        raise ValueError(f"Expected exactly one terminal synthesis task, found {terminals}")
    return outputs[terminals[0]]
