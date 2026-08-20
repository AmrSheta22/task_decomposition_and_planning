from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict


class DynamicDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool
    next_task: str


def dynamic_decomposition(goal: str, llm: BaseChatModel, max_steps: int = 4) -> list[tuple[str, str]]:
    history: list[tuple[str, str]] = []
    for step in range(max_steps):
        observation = "\n".join(f"TASK: {task}\nRESULT: {result}"for task, result in history) or "No tasks have been executed yet."
        decision = llm.with_structured_output(
            DynamicDecision,
            method="json_schema",
        ).invoke([
            ("system", """You are an adaptive planner for the Talenta Recruitment system.Choose the next task based on the observations from previously executed tasks.Use only the available Talenta recruitment capabilities.Do not invent tools or external sources."""),
            ("human", f"""Goal: {goal}
Completed work and observations:
{observation}

Decide the single best next task. Set done to true only when the goal is met.
When done is true, use an empty string for next_task."""),
        ], temperature=0.1)
        if decision.done:
            break
        task = decision.next_task.strip()
        if not task:
            raise ValueError(f"Dynamic planner omitted next_task at step {step + 1}")
        response = llm.invoke([
            ("system","""You are the execution component of the Talenta Recruitment system.Execute the requested sub-task using only the information and capabilities available in the current context.Do not invent tools, external sources, or actions that are not available."""),
            ("human", f"Goal: {goal}\nNext task: {task}\nPrior observations:\n{observation}"),
        ], temperature=0.2)
        result = response.content
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        result = result.strip()
        history.append((task, result))
    return history
