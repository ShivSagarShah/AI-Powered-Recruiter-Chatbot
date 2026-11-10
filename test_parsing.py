from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from src.config import DATA_DIR, JOBS_DIR, RESUMES_DIR
from src.parsers import parse_candidate_resume, parse_job_description


class TestCandidateNameExtraction(TestCase):
    def test_extracts_name_from_resume_heading(self) -> None:
        profile = parse_candidate_resume("# Ada Lovelace\n\nSoftware engineer")
        self.assertEqual(profile.name, "Ada Lovelace")

    def test_skips_contact_and_role_lines_before_name(self) -> None:
        profile = parse_candidate_resume(
            "Curriculum Vitae\n"
            "Senior Software Engineer\n"
            "Name: Grace Hopper\n"
            "grace@example.com\n"
        )
        self.assertEqual(profile.name, "Grace Hopper")

    def test_returns_empty_when_resume_has_no_name(self) -> None:
        profile = parse_candidate_resume(
            "Resume\n"
            "Software Engineer\n"
            "Python and distributed systems\n"
        )
        self.assertEqual(profile.name, "")


def main() -> None:
    jd_files = sorted(JOBS_DIR.glob("*"))
    resume_files = sorted(RESUMES_DIR.glob("*"))

    if not jd_files:
        raise FileNotFoundError(f"No job description files found in {JOBS_DIR}")
    if not resume_files:
        raise FileNotFoundError(f"No resume files found in {RESUMES_DIR}")

    for jd_path in jd_files:
        parsed = parse_job_description(jd_path)
        if not parsed.title:
            raise ValueError(f"Job description missing title: {jd_path.name}")

    for resume_path in resume_files:
        parsed = parse_candidate_resume(resume_path)
        if not parsed.name:
            raise ValueError(f"Candidate resume missing name: {resume_path.name}")

    print(f"Validated {len(jd_files)} job descriptions and {len(resume_files)} resumes successfully.")
    print(f"Dataset root: {DATA_DIR}")


if __name__ == "__main__":
    main()
