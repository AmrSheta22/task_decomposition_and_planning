import asyncio
import random
from types import SimpleNamespace

import pytest
from langchain_core.tools import BaseTool

from planning_lab.algorithms import (
    Environment,
    deterministic_checks,
    dynamic_decomposition,
    execute_plan,
    final_output,
    flatten_lats_tree,
    lats,
    reflexion,
)
from planning_lab.models import EnvironmentFeedback, Plan
from planning_lab.algorithms.decomposition import GeneratedPlan
from planning_lab.algorithms.dynamic_decomposition import DynamicDecision
from planning_lab.algorithms.lats import LATSActionBatch, ValueEstimate
from planning_lab.algorithms.tree_of_thoughts import ThoughtCandidates, ThoughtEvaluation
from langchain_google_genai import ChatGoogleGenerativeAI


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, messages, **kwargs):
        prompt = messages[-1][1]
        self.prompts.append(prompt)
        current = next(
            line.strip() for line in prompt.splitlines() if line.strip().startswith("Current task:")
        )
        return SimpleNamespace(
            content=f"Completed {current} with enough concrete detail for the downstream synthesis task."
        )


def test_dag_order_and_parallel_batches():
    plan = Plan.model_validate({
        "goal": "Prepare a useful launch brief",
        "tasks": [
            {"id": "research", "instruction": "Research the audience", "depends_on": []},
            {"id": "risks", "instruction": "Identify launch risks", "depends_on": []},
            {"id": "brief", "instruction": "Synthesize the launch brief", "depends_on": ["research", "risks"]},
        ],
    })
    assert plan.execution_batches() == [["research", "risks"], ["brief"]]
    assert plan.topological_order()[-1] == "brief"


def test_cycle_is_rejected():
    with pytest.raises(ValueError, match="Cycle detected"):
        Plan.model_validate({
            "goal": "Reject an invalid cyclic plan",
            "tasks": [
                {"id": "a", "instruction": "Perform task alpha", "depends_on": ["b"]},
                {"id": "b", "instruction": "Perform task beta", "depends_on": ["a"]},
            ],
        })


def test_executor_passes_dependency_outputs():
    plan = Plan.model_validate({
        "goal": "Create a concise combined report",
        "tasks": [
            {"id": "a", "instruction": "Collect useful evidence", "depends_on": []},
            {"id": "b", "instruction": "Synthesize all evidence", "depends_on": ["a"]},
        ],
    })
    llm = RecordingLLM()
    outputs = asyncio.run(execute_plan(plan, llm))
    assert "Completed Current task: Collect useful evidence" in llm.prompts[1]
    assert final_output(plan, outputs) == outputs["b"]


class AsyncOnlyTool(BaseTool):
    """Mimics a real MCP tool: only ainvoke works, .invoke() must fail.
    This is exactly what langchain_mcp_adapters' StructuredTool does when
    it wraps a tool from the real MCP server — calling .invoke() on it
    raises NotImplementedError. This test exists to catch a regression
    back to the old sync `tool.invoke(...)` call in execute_plan(), which
    is what broke the real agent run against the live MCP server
    (NotImplementedError: StructuredTool does not support sync invocation).
    """
    name: str = "get_inventory"
    description: str = "Fake stand-in for the real get_inventory MCP tool."

    def _run(self, *args, **kwargs):
        raise NotImplementedError("StructuredTool does not support sync invocation.")

    async def _arun(self, *args, **kwargs):
        return {"item": "Roma Tomatoes", "current_quantity": 4.5}


def test_execute_plan_calls_async_only_tools_without_erroring():
    plan = Plan.model_validate({
        "goal": "Check stock and report it back",
        "tasks": [
            {
                "id": "check_stock",
                "instruction": "Check current Roma Tomatoes stock at branch 2",
                "depends_on": [],
                "tool_name": "get_inventory",
                "tool_args": {"branch_id": 2, "item_name": "Roma Tomatoes"},
            },
        ],
    })
    tool = AsyncOnlyTool()
    # llm=None is safe here: this plan has no reasoning-only tasks, so
    # execute_plan never touches the llm argument — only the tool path runs.
    outputs = asyncio.run(execute_plan(plan, llm=None, tools={"get_inventory": tool}))
    assert "Roma Tomatoes" in outputs["check_stock"]
    assert final_output(plan, outputs) == outputs["check_stock"]



class DynamicDecisionLLM:
    """Fakes a planner that immediately decides to call a real tool, then
    reports done. Mirrors RecordingLLM's shape but for dynamic_decomposition's
    with_structured_output(...).invoke(...) call pattern."""
    class Structured:
        def __init__(self, owner):
            self.owner = owner
            self.calls = 0

        def invoke(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return DynamicDecision(
                    done=False,
                    next_task="Check current Roma Tomatoes stock at branch 2",
                    tool_name="get_inventory",
                    tool_args={"branch_id": 2, "item_name": "Roma Tomatoes"},
                )
            return DynamicDecision(done=True, next_task="")

    def with_structured_output(self, schema, *, method):
        assert method == "json_schema"
        if not hasattr(self, "_structured"):
            self._structured = self.Structured(self)
        return self._structured


def test_dynamic_decomposition_calls_async_only_tools_without_erroring():
    tool = AsyncOnlyTool()
    history = asyncio.run(dynamic_decomposition(
        "Check stock and report it back",
        DynamicDecisionLLM(),
        tools={"get_inventory": tool},
        tool_descriptions="- get_inventory: looks up current stock for an item.",
    ))
    assert len(history) == 1
    assert "Roma Tomatoes" in history[0][1]


def test_grounded_checks_are_deterministic():
    issues = deterministic_checks("Design a phishing awareness workshop", "Too short")
    assert len(issues) >= 2


def good_deliverable() -> str:
    body = " ".join(["security checklist explains structured controls and verification"] * 14)
    return f"# Security Checklist\n- {body}"


class SequencedEnvironment:
    def __init__(self, feedback: list[EnvironmentFeedback]):
        self.feedback = iter(feedback)

    def evaluate(self, state: str) -> EnvironmentFeedback:
        return next(self.feedback)





class ReflexionLLM:
    def __init__(self):
        self.acting_calls = 0
        self.second_trial_saw_memory = False

    def invoke(self, messages, **kwargs):
        system, prompt = messages[0][1], messages[-1][1]
        if "acting agent" in system:
            self.acting_calls += 1
            if self.acting_calls == 1:
                return SimpleNamespace(content="A short security answer.")
            self.second_trial_saw_memory = "I omitted structure" in prompt
            return SimpleNamespace(content=good_deliverable())
        return SimpleNamespace(
            content="I omitted structure and detail; next time I will add a checklist and verification steps."
        )


def test_reflexion_retries_with_bounded_memory():
    llm = ReflexionLLM()
    environment = SequencedEnvironment([
        EnvironmentFeedback(success=False, score=0.3, details=["Random rejection."]),
        EnvironmentFeedback(success=True, score=0.9),
    ])
    result = reflexion(
        "Create a structured security checklist", llm, environment, max_trials=2, memory_size=1
    )
    assert result.success is True
    assert len(result.trials) == 2
    assert result.trials[0].feedback.success is False
    assert result.trials[0].reflection.startswith("I omitted")
    assert llm.second_trial_saw_memory is True
    assert len(result.memory) == 1


class LATSLLM:
    class Structured:
        def __init__(self, owner, schema):
            self.owner = owner
            self.schema = schema

        def invoke(self, messages, **kwargs):
            return self.owner.structured(self.schema)

    def with_structured_output(self, schema, *, method):
        assert method == "function_calling"
        return self.Structured(self, schema)

    def structured(self, schema):
        if schema.__name__ == "LATSActionBatch":
            return schema.model_validate({
                "actions": [
                    {"action": "minimal", "state": "Too short"},
                    {"action": "structured", "state": good_deliverable()},
                ]
            })
        return schema(score=0.8)

    def invoke(self, messages, **kwargs):
        return SimpleNamespace(
            content="This branch failed external length and structure checks; expand with concrete controls."
        )


def test_lats_uses_external_feedback_reflection_and_backpropagation():
    environment = SequencedEnvironment([
        EnvironmentFeedback(success=False, score=0.2, details=["Random rejection."]),
        EnvironmentFeedback(success=True, score=1.0),
    ])
    result = lats(
        "Create a structured security checklist",
        LATSLLM(),
        environment,
        iterations=1,
        n_actions=2,
    )
    assert result.success is True
    assert result.best_score == 1.0
    assert result.root.visits == 2
    assert result.root.children[0].reflections
    tree = flatten_lats_tree(result.root)
    assert len(tree) == 3
    assert tree[1]["feedback"]["success"] is False
    assert tree[2]["feedback"]["success"] is True


@pytest.mark.parametrize(
    "schema",
    [GeneratedPlan, DynamicDecision, ThoughtCandidates, ThoughtEvaluation, LATSActionBatch, ValueEstimate],
)
def test_structured_schemas_bind_with_langchain_gemini(schema):
    chat = ChatGoogleGenerativeAI(google_api_key="test-key", model="test-model")
    runnable = chat.with_structured_output(schema, method="json_schema")
    assert runnable is not None