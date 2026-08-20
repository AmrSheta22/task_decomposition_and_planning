from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from ..models import EnvironmentFeedback
from .environment import Environment


@dataclass
class ReflexionTrial:
    number: int
    attempt: str
    feedback: EnvironmentFeedback
    reflection: str | None = None


@dataclass
class ReflexionResult:
    success: bool
    output: str
    trials: list[ReflexionTrial]
    memory: list[str]


def reflexion(
    task: str,
    llm: BaseChatModel,
    environment: Environment,
    max_trials: int = 3,
    memory_size: int = 3,
) -> ReflexionResult:

    # ============================================================
    # Validate inputs
    # ============================================================

    if not isinstance(task, str) or not task.strip():
        raise ValueError(
            "task must be a non-empty string."
        )

    if max_trials < 1:
        raise ValueError(
            "max_trials must be positive."
        )

    if memory_size < 1:
        raise ValueError(
            "memory_size must be positive."
        )

    memory: list[str] = []
    trials: list[ReflexionTrial] = []

    best_attempt = ""
    best_score = -1.0

    # ============================================================
    # IMPORTANT:
    # Build REAL Talenta context BEFORE the first LLM call.
    # ============================================================

    try:
        grounded_context = (
            environment.get_grounded_context(task)
        )

    except Exception as exc:
        return ReflexionResult(
            success=False,
            output=(
                "Unable to build grounded Talenta context: "
                f"{exc}"
            ),
            trials=[],
            memory=[],
        )

    current_state = ""

    # ============================================================
    # Reflexion loop
    # ============================================================

    for number in range(
        1,
        max_trials + 1,
    ):

        recalled = (
            "\n".join(
                f"- {item}"
                for item in memory[-memory_size:]
            )
            if memory
            else "- No prior reflections."
        )

        previous_attempt = (
            current_state
            if current_state
            else "No previous attempt."
        )

        # ========================================================
        # Generate grounded attempt
        # ========================================================

        response = llm.invoke(
            [
                (
                    "system",
                    """
You are the acting recruitment-review agent for the
Talenta Recruitment system.

You MUST produce the recruitment review using ONLY the
grounded Talenta database context supplied in the user message.

GROUNDING RULES:

1. Never invent:
   - candidate names
   - candidate IDs
   - application IDs
   - job IDs
   - skills
   - experience
   - education
   - recruiter notes
   - interview information
   - application history
   - job requirements

2. Every candidate mentioned must exist in the grounded context.

3. Every application mentioned must belong to the target job.

4. Do NOT say that you lack access to Talenta.

5. Do NOT create fictional examples.

6. Do NOT preserve unsupported facts from a previous attempt.

7. The grounded context is the source of truth.

8. If a requested field is absent from the grounded context,
   write exactly:
   "Not available in the Talenta data."

HIRING POLICY:

- Experience below minimum => REJECT.
- Completely unrelated education => PENDING.
- Calculated skill match below 75% => PENDING.
- Cybersecurity candidates without Linux AND Networking => REJECT.
- Qualified candidates => ADVANCE.
- ADVANCE is NOT ACCEPTED.
- Final ACCEPT/HIRE requires explicit HR Manager confirmation.

VALID AI RECOMMENDATIONS:

- ADVANCE
- PENDING
- REJECT

IMPORTANT:

Evaluate EVERY active applicant in the grounded context.

For every active applicant provide, when available:

- Application ID
- Candidate ID
- Candidate name
- Experience
- Education
- Skills
- Application status
- Recruiter notes
- Interview/application evidence
- Calculated skill match
- Recommendation
- Short grounded reason

Do not omit active applicants.

Do not invent applicants.

Return ONLY the recruitment review.
""",
                ),
                (
                    "human",
                    f"""
TASK:

{task}

============================================================
GROUNDED TALENTA DATABASE CONTEXT
============================================================

{grounded_context}

============================================================
PREVIOUS REFLEXION MEMORY
============================================================

{recalled}

============================================================
PREVIOUS ATTEMPT
============================================================

{previous_attempt}

============================================================

Produce the complete recruitment review.

The grounded Talenta database context is authoritative.

If the previous attempt was wrong, rebuild the answer
from the grounded context.

Do not invent or preserve unsupported information.
""",
                ),
            ],
            temperature=0.2,
        )

        attempt = response.content

        if (
            not isinstance(attempt, str)
            or not attempt.strip()
        ):
            raise RuntimeError(
                "The chat model returned an empty "
                "or unsupported response."
            )

        attempt = attempt.strip()

        # ========================================================
        # CRITICAL:
        # Evaluate THIS NEW attempt.
        # Never evaluate current_state here.
        # ========================================================

        feedback = environment.evaluate(
            attempt,
            task,
        )

        trial = ReflexionTrial(
            number=number,
            attempt=attempt,
            feedback=feedback,
        )

        # ========================================================
        # Track best attempt
        # ========================================================

        if feedback.score > best_score:
            best_score = feedback.score
            best_attempt = attempt

        # ========================================================
        # Success
        # ========================================================

        if feedback.success:

            trials.append(trial)

            return ReflexionResult(
                success=True,
                output=attempt,
                trials=trials,
                memory=memory[-memory_size:],
            )

        # ========================================================
        # Grounded evaluator feedback
        # ========================================================

        feedback_details = (
            "\n".join(
                f"- {item}"
                for item in feedback.details
            )
            if feedback.details
            else "- No detailed feedback was returned."
        )

        # ========================================================
        # Generate Reflexion memory
        # ========================================================

        reflection_response = llm.invoke(
            [
                (
                    "system",
                    """
Generate a concise first-person Reflexion memory.

Rules:

- Start with "I".
- Use ONLY the grounded environment feedback.
- Identify what went wrong.
- State exactly what must change next.
- If an applicant was missing, state that every active
  applicant must be included.
- If a recommendation was wrong, state that it must be
  corrected according to the grounded policy.
- If unsupported information was detected, state that it
  must be removed.
- Do not invent database facts.
- Do not rewrite the recruitment review.
""",
                ),
                (
                    "human",
                    f"""
TASK:

{task}

PREVIOUS ATTEMPT:

{attempt}

GROUNDED ENVIRONMENT SCORE:

{feedback.score}

GROUNDED ENVIRONMENT SUCCESS:

{feedback.success}

GROUNDED ENVIRONMENT FEEDBACK:

{feedback_details}

Explain what I must change in the next trial.
""",
                ),
            ],
            temperature=0.2,
        )

        reflection = reflection_response.content

        if (
            not isinstance(reflection, str)
            or not reflection.strip()
        ):
            raise RuntimeError(
                "The chat model returned an empty "
                "or unsupported reflection."
            )

        reflection = reflection.strip()

        trial.reflection = reflection

        trials.append(trial)

        # ========================================================
        # Store recent reflection
        # ========================================================

        memory.append(reflection)

        if len(memory) > memory_size:
            memory = memory[-memory_size:]

        # ========================================================
        # IMPORTANT:
        # Next iteration sees the latest attempt.
        # ========================================================

        current_state = attempt

    # ============================================================
    # All trials failed.
    #
    # Return the best attempt according to grounded score.
    # ============================================================

    return ReflexionResult(
        success=False,
        output=best_attempt,
        trials=trials,
        memory=memory[-memory_size:],
    )