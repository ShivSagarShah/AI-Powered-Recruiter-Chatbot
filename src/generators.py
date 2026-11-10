from __future__ import annotations

import json
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.matcher import CandidateEvaluation
from src.parsers import clean_text


class InterviewQuestion(BaseModel):
    question: str
    good_answer: str
    red_flag: str
    skill_focus: str | None = None
    gap_focus: str | None = None


class InterviewQuestionSet(BaseModel):
    technical_questions: list[InterviewQuestion] = Field(default_factory=list)
    probing_questions: list[InterviewQuestion] = Field(default_factory=list)


def _normalize_skill(skill: str) -> str:
    return re.sub(r"\s+", " ", (skill or "")).strip().lower()


def _generate_local_questions(candidate_eval: CandidateEvaluation, jd_text: str) -> dict:
    jd_lower = jd_text.lower()
    matched = [skill for skill in candidate_eval.matched_skills if skill]
    missing = [skill for skill in candidate_eval.missing_skills if skill]

    if not matched:
        matched = ["software engineering", "problem solving", "technical collaboration"]
    if not missing:
        missing = ["cross-functional communication", "system design trade-offs", "production ownership"]

    technical = []
    for skill in matched[:4]:
        technical.append(
            {
                "question": f"Describe a project where you used {skill} to drive measurable impact.",
                "good_answer": "They explain the problem, trade-offs, implementation details, and business outcome with metrics.",
                "red_flag": "They rely on buzzwords, cannot explain architecture decisions, or cannot quantify impact.",
                "skill_focus": skill,
            }
        )

    probing = []
    for gap in missing[:3]:
        probing.append(
            {
                "question": f"How have you handled situations where {gap} was a key risk or requirement in a product or system design?",
                "good_answer": "They describe a concrete experience, why the gap mattered, and how they learned or adapted under pressure.",
                "red_flag": "They give vague generalizations or indicate they have never encountered that type of challenge.",
                "gap_focus": gap,
            }
        )

    return {
        "technical_questions": technical,
        "probing_questions": probing,
    }


def _llm_generated_questions(candidate_eval: CandidateEvaluation, jd_text: str) -> dict | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    prompt = (
        "Generate a structured interview plan for a candidate. "
        "Return two arrays: 'technical_questions' and 'probing_questions'. "
        "Each question object must include: 'question', 'good_answer', 'red_flag' and optional 'skill_focus' or 'gap_focus'. "
        "Do not invent experience; base questions on the candidate's matched and missing skills and the job description.\n\n"
        f"CANDIDATE:\n{candidate_eval.model_dump_json(indent=2)}\n\nJOB_DESCRIPTION:\n{clean_text(jd_text)}"
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=api_key,
        )
        result = llm.with_structured_output(InterviewQuestionSet).invoke(prompt)
        return result.model_dump(exclude_none=True)
    except Exception:
        return None


def generate_interview_questions(candidate_eval: CandidateEvaluation, jd_text: str) -> dict:
    """Return a structured set of technical and probing interview questions with good/red-flag answers."""
    if not candidate_eval or not candidate_eval.candidate_name:
        raise ValueError("Candidate evaluation is missing a name.")
    if not jd_text or not str(jd_text).strip():
        raise ValueError("Job description text is empty. Please provide a valid JD.")

    try:
        llm_output = _llm_generated_questions(candidate_eval, jd_text)
        if llm_output:
            return llm_output
    except Exception:
        pass

    return _generate_local_questions(candidate_eval, jd_text)


def _llm_email_draft(candidate_eval: CandidateEvaluation, jd_title: str, email_type: str, custom_details: dict) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    payload = {
        "candidate_name": candidate_eval.candidate_name,
        "jd_title": jd_title,
        "email_type": email_type,
        "fit_level": candidate_eval.fit_level,
        "match_score": candidate_eval.match_score,
        "matched_skills": candidate_eval.matched_skills,
        "missing_skills": candidate_eval.missing_skills,
        "custom_details": custom_details,
    }

    prompt = (
        "Draft a professional recruiter email. "
        "Use only the candidate and job facts supplied below. Do not invent personal details. "
        "Format as clean Markdown with a subject and body.\n\n"
        f"DATA:\n{json.dumps(payload, indent=2)}"
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.25,
            google_api_key=api_key,
        )
        response = llm.invoke(
            "You are a recruiting coordinator. Write concise, professional, factual emails in Markdown. "
            "Never invent missing details.\n\n" + prompt
        )
        content = response.content
        if content:
            return str(content).strip()
    except Exception:
        return None

    return None


def draft_recruiter_email(candidate_eval: CandidateEvaluation, jd_title: str, email_type: str, custom_details: dict) -> str:
    """Create an email draft for a recruiting workflow using the candidate's details and job context."""
    allowed_types = {"interview_invitation", "assessment_instructions", "polite_rejection"}
    if email_type not in allowed_types:
        raise ValueError(f"Unknown email_type '{email_type}'. Allowed: {sorted(allowed_types)}")
    if candidate_eval is None:
        raise ValueError("Candidate evaluation is required to draft an email.")

    try:
        llm_email = _llm_email_draft(candidate_eval, jd_title, email_type, custom_details)
        if llm_email:
            return llm_email
    except Exception:
        pass

    role_name = jd_title or "the role"
    candidate = candidate_eval.candidate_name or "Candidate"
    fit = candidate_eval.fit_level
    score = candidate_eval.match_score

    if email_type == "interview_invitation":
        round_format = custom_details.get("round_format", "technical screening")
        interviewer_name = custom_details.get("interviewer_name", "the hiring team")
        availability = custom_details.get("availability_slots", ["Tuesday 10:00-10:30 AM PT", "Thursday 2:00-2:30 PM PT"])
        availability_text = ", ".join(availability)
        return (
            f"## Subject: Interview Invitation for {role_name}\n\n"
            f"Hi {candidate},\n\n"
            f"Thank you for your interest in the {role_name} opportunity. Based on your experience and a match score of {score}/100, we would like to invite you to a {round_format} round with {interviewer_name}.\n\n"
            f"We particularly noted your strengths in {', '.join(candidate_eval.matched_skills[:3]) if candidate_eval.matched_skills else 'relevant domain experience'}.\n\n"
            f"Please let us know which of the following times work best for you: {availability_text}.\n\n"
            f"Best regards,\n"
            f"Recruiting Team"
        )

    if email_type == "assessment_instructions":
        platform_link = custom_details.get("platform_link", "https://example.com/assessment")
        deadline = custom_details.get("deadline", "within 48 hours")
        guidelines = custom_details.get("guidelines", ["Use your own work, do not share the prompt, and aim for a thoughtful, concise response."])
        guidelines_text = "\n- " + "\n- ".join(guidelines)
        return (
            f"## Subject: Assessment Instructions for {role_name}\n\n"
            f"Hi {candidate},\n\n"
            f"Thank you for progressing in the {role_name} process. As the next step, please complete the assessment using this link: {platform_link}.\n\n"
            f"Please submit your response by {deadline}.\n\n"
            f"Guidelines:{guidelines_text}\n\n"
            f"We appreciate your time and look forward to reviewing your submission.\n\n"
            f"Best,\n"
            f"Recruiting Team"
        )

    return (
        f"## Subject: Update on Your Application for {role_name}\n\n"
        f"Hi {candidate},\n\n"
        f"Thank you for your interest in the {role_name} opportunity and for taking the time to speak with our team. After careful review, we have decided not to move forward with your application at this stage.\n\n"
        f"Your background demonstrates {', '.join(candidate_eval.matched_skills[:3]) if candidate_eval.matched_skills else 'relevant experience'}, and we appreciate the effort you invested in the process. We encourage you to consider future opportunities that may be a closer match for your experience.\n\n"
        f"We wish you the best in your search and appreciate your interest in our company.\n\n"
        f"Best regards,\n"
        f"Recruiting Team"
    )


if __name__ == "__main__":
    from src.matcher import CandidateEvaluation

    sample_eval = CandidateEvaluation(
        candidate_name="Amelia Chen",
        match_score=83,
        matched_skills=["python", "fastapi", "postgresql", "aws"],
        missing_skills=["kubernetes", "system design"],
        years_of_experience_est=7.0,
        overall_summary="Strong backend fit with a clear technical background and leadership experience.",
        fit_level="High",
    )

    sample_jd = "Senior Backend Engineer: Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, AWS, and distributed systems."

    questions = generate_interview_questions(sample_eval, sample_jd)
    print("## Interview Questions")
    print(json.dumps(questions, indent=2))

    email = draft_recruiter_email(
        sample_eval,
        "Senior Backend Engineer",
        "interview_invitation",
        {
            "round_format": "technical panel",
            "interviewer_name": "Priya Shah",
            "availability_slots": ["Tue 3:00 PM PT", "Wed 10:30 AM PT"],
        },
    )
    print("\n## Email Draft")
    print(email)
