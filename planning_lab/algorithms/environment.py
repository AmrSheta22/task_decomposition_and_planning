from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from ..models import EnvironmentFeedback


# ============================================================
# Resolve the real MCP server directory
# ============================================================

CURRENT_FILE = Path(__file__).resolve()


def _find_mcp_server(start: Path) -> Path:
    """
    Find the real mcp_server directory by walking upward.
    """

    current = start.parent

    for directory in [current, *current.parents]:
        candidate = directory / "mcp_server"

        if candidate.is_dir():
            return candidate

    raise RuntimeError(
        "MCP server directory not found.\n"
        f"Current environment file: {CURRENT_FILE}\n"
        "Searched all parent directories for 'mcp_server'."
    )


MCP_SERVER_DIR = _find_mcp_server(CURRENT_FILE)

if str(MCP_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_SERVER_DIR))


from db import get_connection


class Environment:
    """
    Grounded evaluator for Talenta Recruitment.

    The SQLite database is the source of truth.

    This class is responsible for:

    1. Finding the target job from the task.
    2. Loading the real job.
    3. Loading all active applicants.
    4. Loading candidate skills.
    5. Loading required job skills.
    6. Calculating skill match.
    7. Applying Talenta hiring policy.
    8. Building grounded context for the LLM.
    9. Evaluating the NEW generated attempt.
    """

    def __init__(self, success_threshold: float = 0.6):
        if not 0.0 <= success_threshold <= 1.0:
            raise ValueError(
                "success_threshold must be between 0 and 1."
            )

        self.success_threshold = success_threshold

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _normalize(value: Any) -> str:
        if value is None:
            return ""

        return re.sub(
            r"\s+",
            " ",
            str(value).lower(),
        ).strip()

    @staticmethod
    def _format_skills(skills: set[str]) -> str:
        if not skills:
            return "Not available in the Talenta data."

        return ", ".join(sorted(skills))

    # ============================================================
    # Database
    # ============================================================

    def _load_jobs(self) -> list[dict]:
        conn = get_connection()

        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    job_id,
                    title,
                    department,
                    required_degree,
                    min_experience,
                    status
                FROM Jobs
                ORDER BY job_id
                """
            )

            return [
                dict(row)
                for row in cursor.fetchall()
            ]

        finally:
            conn.close()

    def _find_target_job(
        self,
        task: str,
        jobs: list[dict],
    ) -> dict | None:

        normalized_task = self._normalize(task)

        # Exact title match.
        for job in jobs:
            title = self._normalize(job["title"])

            if title and title in normalized_task:
                return job

        # Token fallback.
        for job in jobs:
            title = self._normalize(job["title"])

            tokens = [
                token
                for token in re.findall(
                    r"[a-z0-9]+",
                    title,
                )
                if len(token) > 2
            ]

            if tokens and all(
                token in normalized_task
                for token in tokens
            ):
                return job

        return None

    # ============================================================
    # Skills
    # ============================================================

    def _get_candidate_skills(
        self,
        cursor,
        candidate_id: int,
    ) -> set[str]:

        cursor.execute(
            """
            SELECT skill
            FROM CandidateSkills
            WHERE candidate_id = ?
            """,
            (candidate_id,),
        )

        return {
            self._normalize(row["skill"])
            for row in cursor.fetchall()
            if row["skill"] is not None
        }

    def _get_job_skills(
        self,
        cursor,
        job_id: int,
    ) -> set[str]:

        cursor.execute(
            """
            SELECT skill
            FROM JobSkills
            WHERE job_id = ?
            """,
            (job_id,),
        )

        return {
            self._normalize(row["skill"])
            for row in cursor.fetchall()
            if row["skill"] is not None
        }

    # ============================================================
    # Applications
    # ============================================================

    def _load_active_applications(
        self,
        job_id: int,
    ) -> list[dict]:

        conn = get_connection()

        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    a.application_id,
                    a.candidate_id,
                    a.job_id,
                    a.status,
                    a.match_score,
                    a.recruiter_notes,
                    a.created_at,

                    c.name AS candidate_name,
                    c.email,
                    c.experience_years,
                    c.education,

                    j.title AS job_title,
                    j.department,
                    j.required_degree,
                    j.min_experience,
                    j.status AS job_status

                FROM Applications a

                JOIN Candidates c
                    ON a.candidate_id = c.candidate_id

                JOIN Jobs j
                    ON a.job_id = j.job_id

                WHERE a.job_id = ?
                  AND UPPER(a.status) = 'PENDING'
                  AND UPPER(j.status) = 'OPEN'

                ORDER BY a.application_id
                """,
                (job_id,),
            )

            applications: list[dict] = []

            for row in cursor.fetchall():
                application = dict(row)

                application["candidate_skills"] = (
                    self._get_candidate_skills(
                        cursor,
                        application["candidate_id"],
                    )
                )

                application["job_skills"] = (
                    self._get_job_skills(
                        cursor,
                        application["job_id"],
                    )
                )

                applications.append(application)

            return applications

        finally:
            conn.close()

    # ============================================================
    # Policy calculations
    # ============================================================

    @staticmethod
    def _calculate_match_percentage(
        candidate_skills: set[str],
        job_skills: set[str],
    ) -> float:

        if not job_skills:
            return 0.0

        overlap = candidate_skills & job_skills

        return (
            len(overlap)
            / len(job_skills)
        ) * 100.0

    @staticmethod
    def _education_matches(
        candidate_education: str | None,
        required_degree: str | None,
    ) -> bool:

        if not candidate_education or not required_degree:
            return False

        education = str(
            candidate_education
        ).strip().lower()

        required = str(
            required_degree
        ).strip().lower()

        if education == required:
            return True

        cs_degrees = {
            "computer science",
            "computer engineering",
            "software engineering",
            "information systems",
            "information technology",
        }

        if (
            education in cs_degrees
            and required in cs_degrees
        ):
            return True

        data_degrees = {
            "data science",
            "artificial intelligence",
        }

        if (
            education in data_degrees
            and required in data_degrees
        ):
            return True

        return False

    def _expected_decision(
        self,
        application: dict,
    ) -> tuple[str, list[str], float]:

        reasons: list[str] = []

        candidate_skills = application[
            "candidate_skills"
        ]

        job_skills = application[
            "job_skills"
        ]

        match_percentage = (
            self._calculate_match_percentage(
                candidate_skills,
                job_skills,
            )
        )

        experience = int(
            application["experience_years"] or 0
        )

        minimum_experience = int(
            application["min_experience"] or 0
        )

        education_ok = (
            self._education_matches(
                application["education"],
                application["required_degree"],
            )
        )

        department = self._normalize(
            application["department"]
        )

        # --------------------------------------------------------
        # Policy 1
        # Experience below minimum => REJECT
        # --------------------------------------------------------

        if experience < minimum_experience:
            reasons.append(
                f"experience {experience} years is below "
                f"the required {minimum_experience} years"
            )

            return (
                "REJECT",
                reasons,
                match_percentage,
            )

        # --------------------------------------------------------
        # Policy 2
        # Cybersecurity without Linux AND Networking => REJECT
        # --------------------------------------------------------

        if department == "cybersecurity":

            has_linux = "linux" in candidate_skills
            has_networking = "networking" in candidate_skills

            if not has_linux and not has_networking:
                reasons.append(
                    "cybersecurity candidate lacks both "
                    "Linux and Networking"
                )

                return (
                    "REJECT",
                    reasons,
                    match_percentage,
                )

        # --------------------------------------------------------
        # Policy 3
        # Unrelated education => PENDING
        # --------------------------------------------------------

        if not education_ok:
            reasons.append(
                "education does not closely match "
                "the required degree"
            )

            return (
                "PENDING",
                reasons,
                match_percentage,
            )

        # --------------------------------------------------------
        # Policy 4
        # Skill match below 75% => PENDING
        # --------------------------------------------------------

        if match_percentage < 75.0:
            reasons.append(
                f"calculated skill match is below 75% "
                f"({match_percentage:.1f}%)"
            )

            return (
                "PENDING",
                reasons,
                match_percentage,
            )

        # --------------------------------------------------------
        # Qualified => ADVANCE
        # Never directly ACCEPT/HIRE.
        # --------------------------------------------------------

        reasons.append(
            f"meets grounded requirements with "
            f"{match_percentage:.1f}% calculated skill match"
        )

        return (
            "ADVANCE",
            reasons,
            match_percentage,
        )

    # ============================================================
    # Grounded context
    # ============================================================

    def get_grounded_context(
        self,
        task: str,
    ) -> str:
        """
        Build a complete grounded context from the real DB.

        This method is intentionally public because reflexion.py
        uses it before asking the LLM to generate an answer.

        The LLM receives only facts returned by this method.
        """

        if not isinstance(task, str) or not task.strip():
            raise ValueError(
                "task must be a non-empty string."
            )

        jobs = self._load_jobs()

        if not jobs:
            raise RuntimeError(
                "No jobs were found in the Talenta database."
            )

        target_job = self._find_target_job(
            task,
            jobs,
        )

        if target_job is None:
            raise RuntimeError(
                "Could not identify a real Talenta job "
                "from the task."
            )

        if (
            self._normalize(target_job["status"])
            != "open"
        ):
            raise RuntimeError(
                f"Target job '{target_job['title']}' "
                f"(job_id={target_job['job_id']}) "
                "is not OPEN."
            )

        applications = self._load_active_applications(
            target_job["job_id"]
        )

        lines: list[str] = []

        lines.append(
            "GROUNDED TALENTA DATABASE CONTEXT"
        )

        lines.append(
            "================================="
        )

        lines.append(
            f"Target Job ID: {target_job['job_id']}"
        )

        lines.append(
            f"Target Job Title: {target_job['title']}"
        )

        lines.append(
            f"Department: {target_job['department']}"
        )

        lines.append(
            "Required Degree: "
            f"{target_job['required_degree']}"
        )

        lines.append(
            "Minimum Experience: "
            f"{target_job['min_experience']} years"
        )

        lines.append(
            f"Job Status: {target_job['status']}"
        )

        # Get job skills separately because the job may have
        # zero applications.
        conn = get_connection()

        try:
            cursor = conn.cursor()

            job_skills = self._get_job_skills(
                cursor,
                target_job["job_id"],
            )

        finally:
            conn.close()

        lines.append(
            "Required Skills: "
            f"{self._format_skills(job_skills)}"
        )

        lines.append("")
        lines.append("ACTIVE APPLICATIONS")
        lines.append("===================")

        if not applications:
            lines.append(
                "No active PENDING applications were found "
                "for this OPEN job."
            )

            return "\n".join(lines)

        for index, application in enumerate(
            applications,
            start=1,
        ):

            expected, reasons, match_percentage = (
                self._expected_decision(application)
            )

            lines.append("")
            lines.append(
                f"Applicant #{index}"
            )

            lines.append(
                f"Application ID: "
                f"{application['application_id']}"
            )

            lines.append(
                f"Candidate ID: "
                f"{application['candidate_id']}"
            )

            lines.append(
                f"Candidate Name: "
                f"{application['candidate_name']}"
            )

            lines.append(
                f"Experience: "
                f"{application['experience_years']} years"
            )

            lines.append(
                f"Education: "
                f"{application['education']}"
            )

            lines.append(
                "Skills: "
                f"{self._format_skills(application['candidate_skills'])}"
            )

            lines.append(
                f"Application Status: "
                f"{application['status']}"
            )

            recruiter_notes = (
                application["recruiter_notes"]
            )

            if recruiter_notes:
                lines.append(
                    f"Recruiter Notes: "
                    f"{recruiter_notes}"
                )
            else:
                lines.append(
                    "Recruiter Notes: "
                    "Not available in the Talenta data."
                )

            lines.append(
                f"Calculated Skill Match: "
                f"{match_percentage:.1f}%"
            )

            lines.append(
                f"Grounded Expected Recommendation: "
                f"{expected}"
            )

            lines.append(
                "Grounded Reason: "
                f"{'; '.join(reasons)}"
            )

        lines.append("")
        lines.append(
            "IMPORTANT: The information above is the "
            "grounded Talenta database evidence."
        )

        return "\n".join(lines)

    # ============================================================
    # Application extraction
    # ============================================================

    def _extract_application_sections(
        self,
        state: str,
        applications: list[dict],
    ) -> dict[int, str]:

        normalized_state = self._normalize(state)

        occurrences: list[tuple[int, int]] = []

        for application in applications:

            application_id = (
                application["application_id"]
            )

            candidate_id = (
                application["candidate_id"]
            )

            candidate_name = self._normalize(
                application["candidate_name"]
            )

            patterns = [
                (
                    rf"\bapplication\s*"
                    rf"(?:id\s*)?"
                    rf"[:#-]?\s*"
                    rf"{re.escape(str(application_id))}"
                    rf"\b"
                ),
                (
                    rf"\bcandidate\s*"
                    rf"(?:id\s*)?"
                    rf"[:#-]?\s*"
                    rf"{re.escape(str(candidate_id))}"
                    rf"\b"
                ),
                re.escape(candidate_name),
            ]

            positions: list[int] = []

            for pattern in patterns:
                for match in re.finditer(
                    pattern,
                    normalized_state,
                ):
                    positions.append(
                        match.start()
                    )

            if positions:
                occurrences.append(
                    (
                        min(positions),
                        application_id,
                    )
                )

        occurrences.sort()

        sections: dict[int, str] = {}

        for index, (
            start,
            application_id,
        ) in enumerate(occurrences):

            if index + 1 < len(occurrences):
                end = occurrences[index + 1][0]
            else:
                end = len(normalized_state)

            sections[application_id] = (
                normalized_state[start:end]
            )

        return sections

    # ============================================================
    # Recommendation extraction
    # ============================================================

    @staticmethod
    def _recommendation_from_text(
        text: str,
    ) -> str | None:

        normalized = Environment._normalize(text)

        explicit = re.search(
            r"(?:recommendation|decision|"
            r"final recommendation|candidate decision)"
            r"\s*[:\-]\s*"
            r"(advance|reject|rejected|pending|"
            r"manual review|hold|shortlist|shortlisted|"
            r"next stage|final review|"
            r"accept|accepted|hire|hired|"
            r"approve|approved)\b",
            normalized,
        )

        if explicit:

            value = explicit.group(1)

            if value in {
                "reject",
                "rejected",
            }:
                return "REJECT"

            if value in {
                "pending",
                "manual review",
                "hold",
            }:
                return "PENDING"

            if value in {
                "advance",
                "shortlist",
                "shortlisted",
                "next stage",
                "final review",
            }:
                return "ADVANCE"

            if value in {
                "accept",
                "accepted",
                "hire",
                "hired",
                "approve",
                "approved",
            }:
                return "ACCEPT"

        if re.search(
            r"\bshould\s+be\s+rejected\b",
            normalized,
        ):
            return "REJECT"

        if re.search(
            r"\brecommend(?:ed)?\s+"
            r"(?:to\s+)?reject\b",
            normalized,
        ):
            return "REJECT"

        if re.search(
            r"\bshould\s+remain\s+pending\b",
            normalized,
        ):
            return "PENDING"

        if re.search(
            r"\brecommend(?:ed)?\s+"
            r"(?:to\s+)?(?:keep\s+)?pending\b",
            normalized,
        ):
            return "PENDING"

        if re.search(
            r"\bshould\s+(?:be\s+)?advanced\b",
            normalized,
        ):
            return "ADVANCE"

        if re.search(
            r"\brecommend(?:ed)?\s+"
            r"(?:to\s+)?advance\b",
            normalized,
        ):
            return "ADVANCE"

        return None

    # ============================================================
    # Evaluation
    # ============================================================

    def evaluate(
        self,
        state: str,
        task: str | None = None,
    ) -> EnvironmentFeedback:

        if not isinstance(state, str) or not state.strip():
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[
                    "Agent produced an empty result."
                ],
            )

        if not isinstance(task, str) or not task.strip():
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[
                    "Grounded evaluation requires the "
                    "original task."
                ],
            )

        try:
            jobs = self._load_jobs()

        except Exception as exc:
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[
                    "Grounded database evaluation failed "
                    f"while loading jobs: {exc}"
                ],
            )

        if not jobs:
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[
                    "No jobs were found in the Talenta database."
                ],
            )

        target_job = self._find_target_job(
            task,
            jobs,
        )

        if target_job is None:
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[
                    "Could not identify a real Talenta job "
                    "from the task."
                ],
            )

        job_id = target_job["job_id"]
        job_title = target_job["title"]

        if (
            self._normalize(target_job["status"])
            != "open"
        ):
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[
                    f"Target job '{job_title}' "
                    f"(job_id={job_id}) is not OPEN."
                ],
            )

        try:
            applications = (
                self._load_active_applications(
                    job_id
                )
            )

        except Exception as exc:
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[
                    "Grounded database evaluation failed "
                    "while loading applications: "
                    f"{exc}"
                ],
            )

        if not applications:
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[
                    f"No active PENDING applications were found "
                    f"for open job '{job_title}'."
                ],
            )

        sections = (
            self._extract_application_sections(
                state,
                applications,
            )
        )

        if not sections:
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[
                    f"The agent did not reference any real "
                    f"active application for '{job_title}'.",
                    "The answer must use real application IDs, "
                    "candidate IDs, or candidate names.",
                ],
            )

        details: list[str] = []

        passed = 0
        total = len(applications)

        for application in applications:

            application_id = (
                application["application_id"]
            )

            candidate_id = (
                application["candidate_id"]
            )

            candidate_name = (
                application["candidate_name"]
            )

            expected, reasons, match_percentage = (
                self._expected_decision(
                    application
                )
            )

            if application_id not in sections:

                details.append(
                    f"Application {application_id} "
                    f"(candidate {candidate_id}, "
                    f"{candidate_name}) was not evaluated. "
                    f"Ground truth is {expected}."
                )

                continue

            section = sections[application_id]

            recommendation = (
                self._recommendation_from_text(
                    section
                )
            )

            if recommendation is None:

                details.append(
                    f"Application {application_id} "
                    f"({candidate_name}): no explicit "
                    f"recommendation found. "
                    f"Ground truth is {expected}."
                )

                continue

            # ACCEPT is not a valid final AI decision.
            # It is only tolerated when explicit HR confirmation
            # exists and the grounded policy expects ADVANCE.

            if recommendation == "ACCEPT":

                hr_confirmation = bool(
                    re.search(
                        r"\bhr\s+manager\b.*?"
                        r"\b(confirm|confirmation|"
                        r"approved|approval)\b",
                        section,
                    )
                    or re.search(
                        r"\b("
                        r"approved\s+by\s+hr|"
                        r"hr\s+approval|"
                        r"hr\s+confirmation|"
                        r"explicit\s+hr\s+confirmation"
                        r")\b",
                        section,
                    )
                )

                if not hr_confirmation:

                    details.append(
                        f"Application {application_id} "
                        f"({candidate_name}): ACCEPT without "
                        f"explicit HR confirmation."
                    )

                    continue

                if expected != "ADVANCE":

                    details.append(
                        f"Application {application_id} "
                        f"({candidate_name}): ACCEPT conflicts "
                        f"with grounded policy. "
                        f"Expected {expected}."
                    )

                    continue

                passed += 1

                details.append(
                    f"Application {application_id} "
                    f"({candidate_name}): explicit HR-confirmed "
                    f"acceptance is consistent with ADVANCE."
                )

                continue

            if recommendation == expected:

                passed += 1

                details.append(
                    f"Application {application_id} "
                    f"({candidate_name}): correct "
                    f"{expected}. Skill match "
                    f"{match_percentage:.1f}%."
                )

            else:

                details.append(
                    f"Application {application_id} "
                    f"({candidate_name}): expected "
                    f"{expected}, but agent recommended "
                    f"{recommendation}. "
                    f"Skill match {match_percentage:.1f}%. "
                    f"Reason: {', '.join(reasons)}."
                )

        score = passed / total
        success = score >= self.success_threshold

        details.insert(
            0,
            (
                f"Grounded evaluation "
                f"{'passed' if success else 'failed'} "
                f"for '{job_title}': "
                f"{passed}/{total} active applications "
                f"matched database-derived rules."
            ),
        )

        return EnvironmentFeedback(
            success=success,
            score=round(score, 4),
            details=details,
        )