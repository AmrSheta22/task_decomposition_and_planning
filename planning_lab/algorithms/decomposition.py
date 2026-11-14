from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict
from langchain_core.tools import BaseTool
from ..models import Plan


PLANNER_SYSTEM = """You are a careful task-decomposition planner.
Produce a small executable DAG, not a prose checklist. Every task must make a concrete
contribution to the goal. Independent research or analysis tasks should be parallel.
The plan must end with exactly one synthesis task depending on every necessary branch."""


class PlannedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    instruction: str
    depends_on: list[str]
    tool_name: str | None = None
    tool_args: dict = {}

class GeneratedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: list[PlannedTask]


def decompose_goal(
    goal: str,
    llm: BaseChatModel,
    tool_descriptions: str = "",
) -> Plan:
    tools_block = (
        f"\n\nReal tools available for tool-shaped tasks (set tool_name and tool_args when a task "
        f"is a direct lookup or action rather than something requiring reasoning):\n{tool_descriptions}"
        if tool_descriptions
        else "\n\nNo real tools are available; leave tool_name unset for every task."
    )
    generated = llm.with_structured_output(
        GeneratedPlan,
        method="json_schema",
    ).invoke([
        ("system", PLANNER_SYSTEM + tools_block),
        ("human", f"""Decompose this goal into 3-6 tasks: {goal!r}
Use short task ids such as t1. Dependencies may refer only to tasks in the plan.
Preserve the supplied goal exactly in the plan's goal field.
Set tool_name only when a task is a direct data lookup or action a real tool already
handles — never set it for tasks that require judgment, comparison, or synthesis."""),
    ], temperature=0.1)
    payload = generated.model_dump()
    payload["goal"] = goal
    return Plan.model_validate(payload)


async def execute_plan(
    plan: Plan,
    llm: BaseChatModel,
    tools: dict[str, "BaseTool"] | None = None,
    max_workers: int = 4,
) -> dict[str, str]:
    tools = tools or {}
    outputs: dict[str, str] = {}
    for batch in plan.execution_batches():
        prompts: dict[str, str] = {}
        tool_tasks: dict[str, str] = {}  # task_id -> tool_name, for tasks that skip the LLM entirely
        for task_id in batch:
            task = plan.task(task_id)
            if task.tool_name and task.tool_name in tools:
                tool_tasks[task_id] = task.tool_name
                continue
            context = "\n\n".join(
                f"OUTPUT FROM {dependency}:\n{outputs[dependency]}"
                for dependency in task.depends_on
            ) or "No prerequisite outputs."
            prompts[task_id] = f"""Overall goal: {plan.goal}
                Current task: {task.instruction}
                Prerequisite outputs:
                {context}
                Complete only the current task. Be concrete and concise. Do not invent sources."""

        # Real tool calls first — deterministic, no LLM guessing involved.
        # NOTE: MCP tools from langchain_mcp_adapters only implement async
        # execution (ainvoke). Calling .invoke() on them raises
        # NotImplementedError: StructuredTool does not support sync invocation.
        for task_id, tool_name in tool_tasks.items():
            task = plan.task(task_id)
            tool = tools[tool_name]
            result = await tool.ainvoke(task.tool_args)
            outputs[task_id] = str(result).strip()

        # Reasoning-only tasks go through the LLM exactly as before.
        if prompts:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(prompts))) as pool:
                futures = {
                    pool.submit(
                        llm.invoke,
                        [
                            ("system", "You execute one node in a validated task DAG."),
                            ("human", prompt),
                        ],
                        temperature=0.2,
                    ): task_id
                    for task_id, prompt in prompts.items()
                }
                for future in as_completed(futures):
                    content = future.result().content
                    if not isinstance(content, str) or not content.strip():
                        raise RuntimeError("The chat model returned an empty or unsupported response")
                    outputs[futures[future]] = content.strip()
    return outputs

def final_output(plan: Plan, outputs: dict[str, str]) -> str:
    terminals = plan.terminal_tasks()
    if len(terminals) != 1:
        raise ValueError(f"Expected exactly one terminal synthesis task, found {terminals}")
    return outputs[terminals[0]]