import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field


def deterministic_checks(goal: str, draft: str) -> list[str]:
    issues: list[str] = []
    if len(draft.split()) < 80:
        issues.append("The deliverable is under 80 words and is probably incomplete.")
    goal_terms = {
        word.lower()
        for word in re.findall(r"[A-Za-z]{5,}", goal)
        if word.lower() not in {"create", "design", "write", "build", "about", "using"}
    }
    represented = [term for term in goal_terms if term in draft.lower()]
    if goal_terms and not represented:
        issues.append("The output contains none of the goal's significant terms.")
    if not re.search(r"(^|\n)(#{1,3}\s+|\d+[.)]\s+|[-*]\s+)", draft):
        issues.append("The deliverable has no visible structure (headings or list items).")
    return issues


@dataclass
class ReflectionResult:
    draft: str
    critique: str
    revised: str
    grounded_issues: list[str]


class CritiqueResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "REVISE"]
    issues: list[str] = Field(default_factory=list)


def reflect_and_refine(goal: str, draft: str, llm: BaseChatModel) -> ReflectionResult:
    grounded = deterministic_checks(goal, draft)
    grounded_report = "\n".join(f"- {issue}" for issue in grounded) or "- Deterministic checks passed."
    critique = llm.with_structured_output(
        CritiqueResult,
        method="json_schema",
    ).invoke([
        ("system", "You are a separate critic. Judge against the rubric; do not rewrite the draft."),
        ("human", f"""Goal: {goal}
Rubric: correctness, completeness, internal consistency, and instruction adherence.
External deterministic checks:
{grounded_report}

Draft:
{draft}

Return verdict PASS only if the draft satisfies the rubric and the deterministic checks.
Otherwise set verdict to REVISE and list the concrete issues."""),
    ], temperature=0.2)
    critique_text = "\n".join(f"- {issue}" for issue in critique.issues)
    if critique.verdict == "PASS" and not grounded:
        revised = draft
    else:
        response = llm.invoke([
            ("system", "Revise a deliverable using both external checks and an independent critique."),
            ("human", f"Goal: {goal}\n\nDraft:\n{draft}\n\nGrounded checks:\n{grounded_report}\n\nCritique:\n{critique_text or critique.verdict}\n\nReturn only the improved deliverable."),
        ], temperature=0.2)
        revised = response.content
        if not isinstance(revised, str) or not revised.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        revised = revised.strip()
    return ReflectionResult(draft, critique_text or critique.verdict, revised, grounded)
