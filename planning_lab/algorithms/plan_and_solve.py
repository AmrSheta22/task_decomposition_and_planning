from langchain_core.language_models.chat_models import BaseChatModel


def plan_and_solve(question: str, llm: BaseChatModel) -> str:
    response = llm.invoke(
        [
        (
            "system",
            """You use Plan-and-Solve prompting.

First produce one complete PLAN for the request.
The PLAN must contain concrete, ordered steps.

After the PLAN is complete, execute it step by step in the same order.

Rules:
- The plan is generated once only.
- Do not branch into alternative plans.
- Do not backtrack.
- Do not revise or regenerate the plan during execution.
- Complete each step before moving to the next step.
- Use only the available recruitment capabilities and provided evidence.
- Do not invent tools, data, interview results, or policies.
- Clearly separate PLAN from SOLUTION.""",
        ),
        (
            "human",
            f"""{question}

First understand the request and devise the complete plan.

Then carry out that exact plan step by step.
Do not create a new plan after execution starts.

Clearly separate:

PLAN
- Step 1
- Step 2
- Step 3
- ...

SOLUTION
- Execute Step 1
- Execute Step 2
- Execute Step 3
- ...
- Give the final result based only on the available evidence.""",
        ),
    ], temperature=0.2
)

    if not isinstance(response.content, str) or not response.content.strip():
        raise RuntimeError(
            "The chat model returned an empty or unsupported response"
        )

    return response.content.strip()