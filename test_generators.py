from __future__ import annotations

from src.generators import draft_recruiter_email, generate_interview_questions
from src.matcher import CandidateEvaluation


def main() -> None:
    candidate_eval = CandidateEvaluation(
        candidate_name="Amelia Chen",
        match_score=83,
        matched_skills=["python", "fastapi", "postgresql", "aws"],
        missing_skills=["kubernetes", "system design"],
        years_of_experience_est=7.0,
        overall_summary="Strong backend fit with clear technical depth and leadership experience.",
        fit_level="High",
    )
    jd_text = (
        "Senior Backend Engineer role requiring Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, AWS, "
        "REST APIs, system design, and production reliability."
    )

    questions = generate_interview_questions(candidate_eval, jd_text)
    assert isinstance(questions, dict), "Question set must be a dictionary."
    assert "technical_questions" in questions and "probing_questions" in questions
    assert len(questions["technical_questions"]) >= 3, "At least 3 technical questions are required."
    assert len(questions["technical_questions"]) <= 5, "Technical questions must be capped at 5."
    assert len(questions["probing_questions"]) >= 2, "At least 2 probing questions are required."
    assert len(questions["probing_questions"]) <= 3, "Probing questions must be capped at 3."

    for question_group in (questions["technical_questions"], questions["probing_questions"]):
        for item in question_group:
            assert "question" in item and item["question"].strip(), "Question text is missing."
            assert "good_answer" in item and item["good_answer"].strip(), "Good answer missing."
            assert "red_flag" in item and item["red_flag"].strip(), "Red flag example missing."
            assert "```" not in item["question"], "Question text should be plain markdown, not fenced code blocks."

    email_invite = draft_recruiter_email(
        candidate_eval,
        "Senior Backend Engineer",
        "interview_invitation",
        {
            "round_format": "technical panel",
            "interviewer_name": "Priya Shah",
            "availability_slots": ["Tue 3:00 PM PT", "Wed 10:30 AM PT"],
        },
    )
    email_assessment = draft_recruiter_email(
        candidate_eval,
        "Senior Backend Engineer",
        "assessment_instructions",
        {
            "platform_link": "https://example.com/assessment",
            "deadline": "48 hours from now",
            "guidelines": ["Use your own work", "Do not share prompts", "Keep answers concise"],
        },
    )

    for email in (email_invite, email_assessment):
        assert email.strip().startswith("## Subject:"), "Email must start with a markdown subject heading."
        assert "```" not in email, "Email output must not contain fenced code blocks."
        assert "hallucination" not in email.lower(), "Email should not contain explicit hallucination warnings."

    print("Generator validation passed: interview questions and recruiter emails are formatted and grounded.")


if __name__ == "__main__":
    main()
