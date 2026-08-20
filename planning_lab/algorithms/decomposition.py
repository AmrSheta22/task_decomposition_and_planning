from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from ..models import Plan


PLANNER_SYSTEM = """You are a careful task-decomposition planner.

Produce a small executable DAG for the recruitment task.

The complete plan must be generated in one shot BEFORE any task is executed.
Every task must make a concrete contribution to the goal.

Use only the available Talenta MCP capabilities:
- batch_match_candidates
- analyze_recruiter_note
- simulate_hr_login
- approve_final_hire_with_confirmation
- hiring policy resource: talenta://policies/hiring

Independent research or analysis tasks should be parallel when possible.
Dependencies must be explicit.
The plan must be acyclic and executable in topological order.

The plan must end with exactly one synthesis task depending on every necessary branch.
Do not execute any task while generating the plan.
"""


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
    ).invoke(
        [
            ("system", PLANNER_SYSTEM),
            (
                "human",
                f"""Decompose this goal into 3-8 executable tasks:

{goal!r}

Generate the COMPLETE DAG before execution.

Use short task ids such as t1, t2, t3.
Dependencies may refer only to tasks in the plan.
Independent tasks should be parallel where possible.
The final task must be a single synthesis/recommendation task.

Use only the available Talenta MCP capabilities described in the system instructions.

Preserve the supplied goal exactly in the plan's goal field.""",
            ),
        ],
        temperature=0.1,
    )

    # The caller's goal remains authoritative even if the model paraphrases it.
    payload = generated.model_dump()
    payload["goal"] = goal

    # Plan.model_validate() enforces:
    # - unique task IDs
    # - valid dependencies
    # - no self-dependencies
    # - acyclic DAG
    return Plan.model_validate(payload)


def execute_plan(
    plan: Plan,
    llm: BaseChatModel,
    max_workers: int = 4
) -> dict[str, str]:

    outputs: dict[str, str] = {}

    # Execute the already-generated DAG in topological order.
    for batch in plan.execution_batches():

        prompts: dict[str, str] = {}

        for task_id in batch:

            task = plan.task(task_id)

            context = "\n\n".join(
                f"OUTPUT FROM {dependency}:\n{outputs[dependency]}"
                for dependency in task.depends_on
            ) or "No prerequisite outputs."

            prompts[task_id] = f"""
Overall goal:
{plan.goal}

Current task:
{task.instruction}

Prerequisite outputs:
{context}

Complete only the current task.
Be concrete and concise.
Do not invent sources or results.
"""

        # Tasks in the same topological batch are independent
        # and can therefore execute in parallel.
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(batch))
        ) as pool:

            futures = {
                pool.submit(
                    llm.invoke,
                    [
                        (
                            "system",
                            "You execute one node in a validated task DAG."
                        ),
                        ("human", prompt),
                    ],
                    temperature=0.2,
                ): task_id
                for task_id, prompt in prompts.items()
            }

            for future in as_completed(futures):

                content = future.result().content

                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(
                        "The chat model returned an empty or unsupported response"
                    )

                outputs[futures[future]] = content.strip()

    return outputs


def final_output(
    plan: Plan,
    outputs: dict[str, str]
) -> str:

    terminals = plan.terminal_tasks()

    if len(terminals) != 1:
        raise ValueError(
            f"Expected exactly one terminal synthesis task, found {terminals}"
        )

    return outputs[terminals[0]]
