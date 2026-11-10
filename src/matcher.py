from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Callable, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, PrivateAttr

from src.parsers import clean_text, parse_candidate_resume, parse_job_description

SKILL_TERMS = (
    "agile/scrum",
    "ci/cd",
    "css3",
    "cross-browser compatibility",
    "cypress",
    "git",
    "graphql",
    "html5",
    "javascript",
    "jest",
    "react",
    "react testing library",
    "responsive web design",
    "restful apis",
    "state management",
    "tailwind css",
    "typescript",
)
LLM_EVALUATION_TIMEOUT_SECONDS = 120


class CandidateEvaluation(BaseModel):
    candidate_name: str
    match_score: int = Field(ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    years_of_experience_est: float = Field(ge=0.0)
    overall_summary: str
    fit_level: Literal["High", "Medium", "Low"]
    _fallback_reason: str | None = PrivateAttr(default=None)

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason


CandidateProgressCallback = Callable[
    [int, int, str, Literal["started", "completed", "skipped"], CandidateEvaluation | None],
    None,
]


def _normalize_skill(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def _collect_candidate_signals(resume_text: str) -> tuple[str, set[str], float]:
    profile = parse_candidate_resume(resume_text)
    skills = {_normalize_skill(skill) for skill in profile.skills if skill and skill.strip()}
    candidate_name = profile.name or "Candidate"

    stated_years = [
        float(value)
        for value in re.findall(r"\b(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b", resume_text, flags=re.IGNORECASE)
    ]
    years = max(stated_years, default=0.0)
    if not years:
        experience_dates = re.findall(
            r"\b((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2}|present|current)\b",
            resume_text,
            flags=re.IGNORECASE,
        )
        current_year = datetime.now().year
        durations = [
            max(0, (current_year if end.lower() in {"present", "current"} else int(end)) - int(start))
            for start, end in experience_dates
        ]
        years = float(max(durations, default=0))

    return candidate_name, skills, years


def _collect_jd_signals(jd_text: str) -> tuple[set[str], set[str], str]:
    jd = parse_job_description(jd_text)
    jd_skills = {_normalize_skill(skill) for skill in jd.skills if skill and skill.strip()}
    if not jd_skills:
        relevant_jd_text = "\n".join([jd.summary, *jd.responsibilities, *jd.requirements])
        jd_skills = {
            skill
            for skill in SKILL_TERMS
            if _skill_is_present(skill, relevant_jd_text, set())
        }
    title = jd.title or "Target role"

    return jd_skills, set(), title


def _minimum_required_experience(jd_text: str) -> float | None:
    requirements = parse_job_description(jd_text).requirements
    requirement_text = " ".join(requirements)
    match = re.search(
        r"\b(?:(?:at least|minimum(?: of)?|more than)\s+)?(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b",
        requirement_text,
        flags=re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def _skill_is_present(skill: str, resume_text: str, resume_skills: set[str]) -> bool:
    normalized_skill = _normalize_skill(skill)
    if normalized_skill in resume_skills:
        return True

    normalized_text = re.sub(r"[^\w+#]+", " ", resume_text.lower())
    normalized_skill_text = re.sub(r"[^\w+#]+", " ", normalized_skill)
    if re.search(rf"(?<!\w){re.escape(normalized_skill_text)}(?!\w)", normalized_text):
        return True

    if normalized_skill == "responsive web design":
        return bool(re.search(r"\bresponsive(?:\s+web)?\s+design\b", normalized_text))
    if normalized_skill == "sql":
        return any(
            re.search(rf"(?<!\w){database}(?!\w)", normalized_text)
            for database in ("postgresql", "mysql", "sqlite", "sql server", "mssql")
        )
    if normalized_skill in {"api", "apis", "rest api", "rest apis"}:
        return bool(re.search(r"\brest(?:ful)?\s+apis?\b|\bapis?\b", normalized_text))
    return False


def _heuristic_candidate_evaluation(candidate_name: str, resume_text: str, jd_text: str) -> CandidateEvaluation:
    resume_name, resume_skills, years_exp = _collect_candidate_signals(resume_text)
    jd_skills, _, jd_title = _collect_jd_signals(jd_text)

    matched_skills = sorted(
        (skill for skill in jd_skills if _skill_is_present(skill, resume_text, resume_skills)),
        key=lambda x: x,
    )
    missing_skills = sorted(jd_skills - set(matched_skills))

    if not jd_skills:
        skill_score = 60.0
    else:
        skill_score = len(matched_skills) / len(jd_skills) * 70

    title_text = (resume_name + " " + resume_text).lower()
    jd_title_lower = jd_title.lower()
    distinctive_title_terms = [
        term for term in re.findall(r"[a-z]+", jd_title_lower)
        if term not in {"senior", "junior", "lead", "staff", "principal", "engineer", "manager", "specialist"}
    ]
    role_score = 10.0 if distinctive_title_terms and any(term in title_text for term in distinctive_title_terms) else 0.0

    minimum_years = _minimum_required_experience(jd_text)
    if minimum_years:
        experience_score = 20.0 * min(years_exp / minimum_years, 1.0)
    else:
        experience_score = 20.0 * min(years_exp / 10.0, 1.0)
    score = int(round(min(100.0, skill_score + experience_score + role_score)))

    if score >= 75:
        fit_level = "High"
    elif score >= 45:
        fit_level = "Medium"
    else:
        fit_level = "Low"

    summary = (
        f"{candidate_name} matches {len(matched_skills)} of the listed skills for the {jd_title} role, "
        f"with {years_exp:.1f} years of relevant experience."
    )

    return CandidateEvaluation(
        candidate_name=candidate_name,
        match_score=score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        years_of_experience_est=round(years_exp, 1),
        overall_summary=summary,
        fit_level=fit_level,
    )


def _llm_candidate_evaluation(resume_text: str, jd_text: str) -> CandidateEvaluation | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    prompt = (
        "Evaluate this candidate against the given job description. "
        "Use the provided structured output schema and do not include commentary. "
        "Focus on skills overlap, missing requirements, estimated experience, and overall fit.\n\n"
        f"RESUME:\n{clean_text(resume_text)}\n\nJOB DESCRIPTION:\n{clean_text(jd_text)}"
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.1,
        google_api_key=api_key,
        api_version="v1beta",
        request_timeout=LLM_EVALUATION_TIMEOUT_SECONDS,
        max_retries=0,
    )
    structured_llm = llm.with_structured_output(CandidateEvaluation)
    return CandidateEvaluation.model_validate(structured_llm.invoke(prompt))


def _evaluate_candidate_with_fallback(
    candidate_name: str,
    resume_text: str,
    jd_text: str,
) -> CandidateEvaluation:
    candidate_name = parse_candidate_resume(resume_text).name or candidate_name
    fallback_reason = "AI evaluation did not return a result."
    try:
        llm_result = _llm_candidate_evaluation(resume_text, jd_text)
    except Exception as exc:
        fallback_reason = f"AI evaluation failed ({type(exc).__name__}): {exc}"
    else:
        if llm_result is not None:
            llm_result.candidate_name = candidate_name
            return llm_result
        if not os.getenv("GEMINI_API_KEY"):
            fallback_reason = "GEMINI_API_KEY is not configured."

    evaluation = _heuristic_candidate_evaluation(candidate_name, resume_text, jd_text)
    evaluation._fallback_reason = fallback_reason
    return evaluation


def evaluate_candidate(resume_text: str, jd_text: str) -> CandidateEvaluation:
    """Score a candidate against a job description using structured LLM output when available."""
    if not resume_text or not str(resume_text).strip():
        raise ValueError("Resume text is empty. Please provide candidate content before evaluation.")
    if not jd_text or not str(jd_text).strip():
        raise ValueError("Job description text is empty. Please provide the JD before evaluation.")

    try:
        return _evaluate_candidate_with_fallback("Candidate", resume_text, jd_text)
    except Exception as exc:
        raise ValueError(f"Candidate evaluation failed: {exc}") from exc


def _normalize_candidate_entry(entry: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(entry, dict):
        raise TypeError("Each resume item must be a dictionary.")

    candidate_name = str(
        entry.get("candidate_name")
        or entry.get("name")
        or entry.get("candidateName")
        or "Candidate"
    )

    resume_text = (
        entry.get("resume_text")
        or entry.get("cv_text")
        or entry.get("text")
        or entry.get("content")
        or ""
    )

    if not str(resume_text).strip() and "path" in entry:
        resume_path = entry["path"]
        resume_text = clean_text(open(resume_path, "r", encoding="utf-8", errors="replace").read())

    return candidate_name, str(resume_text)


def rank_candidates(
    resumes: list[dict],
    jd_text: str,
    progress_callback: CandidateProgressCallback | None = None,
) -> list[CandidateEvaluation]:
    if not jd_text or not str(jd_text).strip():
        raise ValueError("Job description text is empty. Please provide a valid JD before ranking.")
    if not resumes:
        raise ValueError("No resumes were provided for ranking.")

    evaluations: list[CandidateEvaluation] = []
    for index, resume in enumerate(resumes, start=1):
        candidate_name = "Candidate"
        try:
            candidate_name, resume_text = _normalize_candidate_entry(resume)
            if not resume_text.strip():
                if progress_callback:
                    progress_callback(index, len(resumes), candidate_name, "skipped", None)
                continue

            candidate_name = parse_candidate_resume(resume_text).name or candidate_name
            if progress_callback:
                progress_callback(index, len(resumes), candidate_name, "started", None)
            evaluation = _evaluate_candidate_with_fallback(candidate_name, resume_text, jd_text)
            evaluations.append(evaluation)
            if progress_callback:
                progress_callback(index, len(resumes), evaluation.candidate_name, "completed", evaluation)
        except Exception:
            if progress_callback:
                progress_callback(index, len(resumes), candidate_name, "skipped", None)
            continue

    if not evaluations:
        raise ValueError("No valid candidate resumes remained after parsing and validation.")

    return sorted(evaluations, key=lambda item: (-item.match_score, item.candidate_name.lower()))


if __name__ == "__main__":
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent.parent
    jd_path = base_dir / "data" / "jds" / "senior_backend_engineer.md"
    jd_text = clean_text(jd_path.read_text(encoding="utf-8", errors="replace"))

    resume_entries = []
    for resume_path in sorted((base_dir / "data" / "resumes").glob("*")):
        resume_entries.append({
            "candidate_name": resume_path.stem.replace("_", " ").title(),
            "resume_text": clean_text(resume_path.read_text(encoding="utf-8", errors="replace")),
        })

    ranked = rank_candidates(resume_entries, jd_text)
    for index, result in enumerate(ranked, start=1):
        print(f"{index}. {result.candidate_name} -> {result.match_score}/100 ({result.fit_level})")
        print(f"   matched: {', '.join(result.matched_skills) if result.matched_skills else 'none'}")
        print(f"   missing: {', '.join(result.missing_skills) if result.missing_skills else 'none'}")
