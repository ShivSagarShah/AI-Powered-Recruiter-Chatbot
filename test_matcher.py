from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src.config import JOBS_DIR, RESUMES_DIR
from src.matcher import (
    CandidateEvaluation,
    _collect_candidate_signals,
    _collect_jd_signals,
    _heuristic_candidate_evaluation,
    LLM_EVALUATION_TIMEOUT_SECONDS,
    _llm_candidate_evaluation,
    _minimum_required_experience,
    rank_candidates,
)
from src.parsers import clean_text, parse_job_description


class TestCandidateEvaluationFallback(TestCase):
    def test_plain_text_frontend_jd_and_resume_score_as_high_fit(self) -> None:
        jd_text = """JOB TITLE: Senior Frontend Developer
COMPANY: TechFlow Solutions
LOCATION: Remote
JOB SUMMARY:
Build responsive, high-performance web applications with designers and backend engineers.
RESPONSIBILITIES:
- Develop features using React, TypeScript, and Tailwind CSS.
- Integrate RESTful APIs and GraphQL endpoints.
- Write tests using Jest and React Testing Library.
REQUIREMENTS:
- 5+ years of professional frontend development experience.
- Strong proficiency in JavaScript, TypeScript, React, and modern state management.
- Deep understanding of HTML5, CSS3, and Responsive Web Design.
- Experience with Jest, Cypress, or React Testing Library.
- Familiarity with Git, CI/CD, and Agile/Scrum workflows.
- Bachelor's degree in Computer Science."""
        resume_text = """ALEX MORGAN
alex.morgan.dev@email.com
SUMMARY:
Senior Frontend Developer with over 6 years of experience building high-performance web applications.
SKILLS:
- JavaScript, TypeScript, HTML5, CSS3
- React, Tailwind CSS, GraphQL
- Jest, React Testing Library, Cypress, Git
- Agile/Scrum, CI/CD, Responsive Design, RESTful APIs
PROFESSIONAL EXPERIENCE:
Senior Frontend Developer (2021 – Present)
- Integrated GraphQL APIs and optimized state management.
- Led junior developers and code reviews."""

        parsed_jd = parse_job_description(jd_text)
        result = _heuristic_candidate_evaluation("Alex Morgan", resume_text, jd_text)

        self.assertEqual(parsed_jd.title, "Senior Frontend Developer")
        self.assertEqual(len(parsed_jd.responsibilities), 3)
        self.assertEqual(len(parsed_jd.requirements), 6)
        self.assertGreaterEqual(result.match_score, 90)
        self.assertEqual(result.fit_level, "High")
        self.assertEqual(result.years_of_experience_est, 6)

    def test_jd_skills_are_not_mixed_with_requirement_sentences(self) -> None:
        jd_text = clean_text((JOBS_DIR / "senior_backend_engineer.md").read_text(encoding="utf-8"))
        parsed_jd = parse_job_description(jd_text)

        self.assertEqual(
            set(parsed_jd.skills),
            {"Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "AWS", "Redis", "Kafka"},
        )
        self.assertIn("5+ years", parsed_jd.requirements[0])
        self.assertNotIn("FastAPI", parsed_jd.requirements)
        self.assertEqual(_collect_jd_signals(jd_text)[0], {
            "python", "fastapi", "postgresql", "docker", "kubernetes", "aws", "redis", "kafka",
        })
        self.assertEqual(_minimum_required_experience(jd_text), 5.0)

    def test_explicit_experience_beats_incomplete_resume_date_range(self) -> None:
        resume_text = clean_text(
            (RESUMES_DIR / "amelia_chen_backend_engineer.md").read_text(encoding="utf-8")
        )

        self.assertEqual(_collect_candidate_signals(resume_text)[2], 7.0)

    def test_meeting_experience_requirement_and_skills_scores_as_strong_fit(self) -> None:
        jd_text = clean_text((JOBS_DIR / "senior_backend_engineer.md").read_text(encoding="utf-8"))
        resume_text = (
            "# Candidate Name\n\n"
            "4 years of experience as a Senior Backend Engineer.\n\n"
            "## Skills\n"
            "- Python\n- FastAPI\n- PostgreSQL\n- Docker\n- Kubernetes\n- AWS\n- Redis\n- Kafka"
        )

        result = _heuristic_candidate_evaluation("Candidate Name", resume_text, jd_text)

        self.assertEqual(result.match_score, 96)
        self.assertEqual(result.fit_level, "High")
        self.assertFalse(result.missing_skills)

    def test_exceeding_four_year_requirement_does_not_reduce_score(self) -> None:
        jd_text = clean_text((JOBS_DIR / "senior_backend_engineer.md").read_text(encoding="utf-8"))
        jd_text = jd_text.replace("5+ years", "4+ years")
        resume_text = (
            "# Candidate Name\n\n"
            "5 years of experience as a Senior Backend Engineer.\n\n"
            "## Skills\n"
            "- Python\n- FastAPI\n- PostgreSQL\n- Docker\n- Kubernetes\n- AWS\n- Redis\n- Kafka"
        )

        result = _heuristic_candidate_evaluation("Candidate Name", resume_text, jd_text)

        self.assertEqual(result.match_score, 100)
        self.assertEqual(result.fit_level, "High")

    def test_llm_evaluation_uses_beta_api_for_structured_output(self) -> None:
        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
            patch("src.matcher.ChatGoogleGenerativeAI") as llm_factory,
        ):
            llm_factory.return_value.with_structured_output.return_value.invoke.return_value = {
                "candidate_name": "Model name",
                "match_score": 90,
                "matched_skills": ["python"],
                "missing_skills": [],
                "years_of_experience_est": 5,
                "overall_summary": "LLM result",
                "fit_level": "High",
            }

            result = _llm_candidate_evaluation("Python developer", "Python role")

        llm_factory.assert_called_once_with(
            model="gemini-3.8-flash",
            temperature=0.1,
            google_api_key="test-key",
            api_version="v1beta",
            request_timeout=LLM_EVALUATION_TIMEOUT_SECONDS,
            max_retries=0,
        )
        self.assertIsInstance(result, CandidateEvaluation)

    def test_rank_candidates_reports_progress_per_resume(self) -> None:
        progress = []
        llm_result = CandidateEvaluation(
            candidate_name="Model name",
            match_score=90,
            years_of_experience_est=5,
            overall_summary="LLM result",
            fit_level="High",
        )
        with patch("src.matcher._llm_candidate_evaluation", return_value=llm_result):
            rank_candidates(
                [
                    {"candidate_name": "Ada", "resume_text": "# Ada Lovelace\n\nResume"},
                    {"candidate_name": "Grace", "resume_text": "# Grace Hopper\n\nResume"},
                ],
                "Job description",
                progress_callback=lambda *event: progress.append(event),
            )

        self.assertEqual(
            [(event[0], event[2], event[3]) for event in progress],
            [
                (1, "Ada Lovelace", "started"),
                (1, "Ada Lovelace", "completed"),
                (2, "Grace Hopper", "started"),
                (2, "Grace Hopper", "completed"),
            ],
        )
        self.assertTrue(all(event[1] == 2 for event in progress))

    def test_llm_timeout_falls_back_to_heuristic_and_reports_timeout(self) -> None:
        progress = []
        with (
            patch("src.matcher._llm_candidate_evaluation", side_effect=TimeoutError("request timed out")),
            patch(
                "src.matcher._heuristic_candidate_evaluation",
                return_value=CandidateEvaluation(
                    candidate_name="Ada Lovelace",
                    match_score=80,
                    years_of_experience_est=5,
                    overall_summary="Heuristic result",
                    fit_level="High",
                ),
            ) as heuristic,
        ):
            result = rank_candidates(
                [{"candidate_name": "Ada", "resume_text": "# Ada Lovelace\n\nResume"}],
                "Job description",
                progress_callback=lambda *event: progress.append(event),
            )[0]

        heuristic.assert_called_once()
        self.assertIn("TimeoutError", result.fallback_reason or "")
        self.assertEqual([event[3] for event in progress], ["started", "completed"])

    def test_rank_candidates_falls_back_to_heuristic_with_reason(self) -> None:
        calls = []

        def fail_llm(resume_text: str, jd_text: str) -> None:
            calls.append("llm")
            raise RuntimeError("model unavailable")

        def heuristic(candidate_name: str, resume_text: str, jd_text: str) -> CandidateEvaluation:
            calls.append("heuristic")
            return CandidateEvaluation(
                candidate_name=candidate_name,
                match_score=50,
                years_of_experience_est=0,
                overall_summary="Heuristic result",
                fit_level="Medium",
            )

        with (
            patch("src.matcher._llm_candidate_evaluation", side_effect=fail_llm),
            patch("src.matcher._heuristic_candidate_evaluation", side_effect=heuristic),
        ):
            results = rank_candidates(
                [{"candidate_name": "Ada", "resume_text": "Resume text"}],
                "Job description",
            )

        self.assertEqual(calls, ["llm", "heuristic"])
        self.assertEqual(results[0].candidate_name, "Ada")
        self.assertEqual(
            results[0].fallback_reason,
            "AI evaluation failed (RuntimeError): model unavailable",
        )

    def test_rank_candidates_uses_llm_result_without_fallback(self) -> None:
        calls = []

        def llm(resume_text: str, jd_text: str) -> CandidateEvaluation:
            calls.append("llm")
            return CandidateEvaluation(
                candidate_name="Model name",
                match_score=90,
                years_of_experience_est=5,
                overall_summary="LLM result",
                fit_level="High",
            )

        def heuristic(candidate_name: str, resume_text: str, jd_text: str) -> CandidateEvaluation:
            calls.append("heuristic")
            raise AssertionError("Heuristic evaluation should not run when LLM succeeds.")

        with (
            patch("src.matcher._llm_candidate_evaluation", side_effect=llm),
            patch("src.matcher._heuristic_candidate_evaluation", side_effect=heuristic),
        ):
            results = rank_candidates(
                [{"candidate_name": "Ada", "resume_text": "Resume text"}],
                "Job description",
            )

        self.assertEqual(calls, ["llm"])
        self.assertEqual(results[0].candidate_name, "Ada")
        self.assertIsNone(results[0].fallback_reason)

    def test_resume_text_name_overrides_filename_for_ranking(self) -> None:
        llm_result = CandidateEvaluation(
            candidate_name="Model name",
            match_score=90,
            years_of_experience_est=5,
            overall_summary="LLM result",
            fit_level="High",
        )
        with patch("src.matcher._llm_candidate_evaluation", return_value=llm_result):
            results = rank_candidates(
                [{
                    "candidate_name": "Uploaded Filename",
                    "resume_text": "# Ada Lovelace\n\nExperienced engineer.",
                }],
                "Job description",
            )

        self.assertEqual(results[0].candidate_name, "Ada Lovelace")


def main() -> None:
    jd_path = JOBS_DIR / "senior_backend_engineer.md"
    jd_text = clean_text(jd_path.read_text(encoding="utf-8", errors="replace"))

    resumes = []
    for resume_path in sorted(RESUMES_DIR.glob("*")):
        resumes.append(
            {
                "candidate_name": resume_path.stem.replace("_", " ").title(),
                "resume_text": clean_text(resume_path.read_text(encoding="utf-8", errors="replace")),
            }
        )

    ranked = rank_candidates(resumes, jd_text)
    print("Top ranked candidates for Senior Backend Engineer:")
    for idx, result in enumerate(ranked, start=1):
        print(f"{idx}. {result.candidate_name} - Score: {result.match_score}/100 - Fit: {result.fit_level}")

    if not ranked:
        raise RuntimeError("Ranking did not produce any candidates for evaluation.")


if __name__ == "__main__":
    main()
