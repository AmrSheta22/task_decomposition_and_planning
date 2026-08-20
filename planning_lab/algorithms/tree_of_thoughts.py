from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field
from ..models import Thought
from collections import deque
class ThoughtCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[str] = Field(min_length=1, max_length=3)


class ThoughtEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    rationale: str



def tree_of_thoughts(
    problem: str,
    llm: BaseChatModel,
    depth: int = 2,
    beam_width: int = 2,
    prune_threshold: float = 0.5,
) -> list[Thought]:

    queue = deque([
        (Thought(state="Start", score=0.5, rationale="root"), 0)
    ])

    results: list[Thought] = []

    while queue:

        parent, level = queue.popleft()

        if level >= depth:
            results.append(parent)
            continue

        generated = llm.with_structured_output(
            ThoughtCandidates,
            method="json_schema",
        ).invoke([
            (
                "system",
                """Generate distinct candidate next steps for
                a Tree-of-Thoughts search over a Talenta
                recruitment problem.

                Use only the available recruitment capabilities:
                batch_match_candidates,
                analyze_recruiter_note,
                simulate_hr_login,
                approve_final_hire_with_confirmation,
                and talenta://policies/hiring.
                """,
            ),
            (
                "human",
                f"""Problem: {problem}

Partial path:
{parent.state}

Propose two distinct promising next steps.
""",
            ),
        ], temperature=0.5)

        children = []

        for state in generated.candidates[:2]:

            judged = llm.with_structured_output(
                ThoughtEvaluation,
                method="json_schema",
            ).invoke([
                (
                    "system",
                    """Evaluate this recruitment branch.
                    Score correctness, feasibility, progress,
                    and policy consistency from 0 to 1.
                    Do not invent tool results.
                    """,
                ),
                (
                    "human",
                    f"""Problem: {problem}

Candidate path:
{parent.state}

Next step:
{state}
""",
                ),
            ], temperature=0.1)

            child = Thought(
                state=f"{parent.state}\nNEXT STEP: {state}",
                score=judged.score,
                rationale=judged.rationale,
            )

            children.append(child)

        
        children = [
            child
            for child in children
            if child.score >= prune_threshold
        ]

        
        children = sorted(
            children,
            key=lambda child: child.score,
            reverse=True,
        )[:beam_width]

        

        for child in children:
            queue.append((child, level + 1))

    return results