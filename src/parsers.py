from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field
from pypdf import PdfReader


class JobDescription(BaseModel):
    title: str = ""
    company: str = ""
    location: str = ""
    summary: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    raw_text: str = ""


class CandidateProfile(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    raw_text: str = ""


def clean_text(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_text_from_pdf(path: str | Path) -> str:
    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_text("\n".join(pages))


def extract_text_from_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix in {".md", ".markdown", ".txt", ".rst"}:
        return clean_text(file_path.read_text(encoding="utf-8", errors="replace"))

    raise ValueError(f"Unsupported file type for parser: {suffix}")


def _extract_heading_value(text: str, heading_name: str) -> str:
    pattern = rf"(?:^|\n)#{1,6}\s*{re.escape(heading_name)}\s*\n+(.+?)(?=\n#{1,6}\s|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return clean_text(match.group(1)).strip("-: ")

    pattern = rf"{re.escape(heading_name)}\s*[:\-]\s*(.+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return clean_text(match.group(1)).strip()

    return ""


def _extract_list_items(section_text: str) -> list[str]:
    items: list[str] = []
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ", "+ ")):
            items.append(clean_text(line[2:]))
        elif re.match(r"^\d+\.\s+", line):
            items.append(clean_text(re.sub(r"^\d+\.\s+", "", line)))
        else:
            items.append(clean_text(line))
    return items


def _extract_section_block(text: str, heading_name: str) -> str:
    lines = text.splitlines()
    heading_pattern = re.compile(rf"^\s*#{{1,6}}\s*{re.escape(heading_name)}\s*#*\s*$", re.IGNORECASE)
    next_heading_pattern = re.compile(r"^\s*#{1,6}\s+")
    label_pattern = re.compile(r"^\s*([A-Z][A-Z0-9 &'()/+-]*):\s*(.*)$")
    normalized_heading = re.sub(r"\s+", " ", heading_name).strip().lower()
    for index, line in enumerate(lines):
        if heading_pattern.match(line):
            section_lines = []
            for section_line in lines[index + 1:]:
                if next_heading_pattern.match(section_line):
                    break
                section_lines.append(section_line)
            return "\n".join(section_lines).strip()

        label_match = label_pattern.match(line)
        if label_match and re.sub(r"\s+", " ", label_match.group(1)).strip().lower() == normalized_heading:
            section_lines = [label_match.group(2)] if label_match.group(2) else []
            for section_line in lines[index + 1:]:
                if next_heading_pattern.match(section_line) or label_pattern.match(section_line):
                    break
                section_lines.append(section_line)
            return "\n".join(section_lines).strip()

    return ""


def parse_job_description(path_or_text: str | Path) -> JobDescription:
    if isinstance(path_or_text, (str, Path)) and Path(path_or_text).exists():
        raw_text = extract_text_from_file(path_or_text)
    else:
        raw_text = clean_text(str(path_or_text))

    title = _extract_heading_value(raw_text, "JOB TITLE")
    first_line = raw_text.splitlines()[0].strip() if raw_text else ""
    if not title and first_line.startswith("#"):
        title = clean_text(first_line.lstrip("#").strip())
    elif not title and re.search(r"[A-Za-z].*Engineer|Scientist|Manager|Analyst|Developer", first_line):
        title = first_line

    summary = (
        _extract_heading_value(raw_text, "JOB SUMMARY")
        or _extract_heading_value(raw_text, "Summary")
        or _extract_heading_value(raw_text, "Overview")
    )
    company = _extract_heading_value(raw_text, "Company")
    location = _extract_heading_value(raw_text, "Location")

    responsibilities = _extract_list_items(_extract_section_block(raw_text, "Responsibilities"))
    requirements = _extract_list_items(_extract_section_block(raw_text, "Requirements"))
    skills = _extract_list_items(_extract_section_block(raw_text, "Skills"))

    if not responsibilities:
        responsibilities = _extract_list_items(_extract_section_block(raw_text, "What you will do"))
    if not requirements:
        requirements = _extract_list_items(_extract_section_block(raw_text, "Must have"))
    if not skills:
        skills = _extract_list_items(_extract_section_block(raw_text, "Core skills"))

    return JobDescription(
        title=title,
        company=company,
        location=location,
        summary=summary,
        responsibilities=responsibilities,
        requirements=requirements,
        skills=skills,
        raw_text=raw_text,
    )


def _extract_candidate_name(text: str) -> str:
    ignored_lines = {
        "cv",
        "curriculum vitae",
        "resume",
        "resume text",
        "professional summary",
        "summary",
        "profile",
        "contact",
        "skills",
        "experience",
        "work experience",
        "education",
    }
    role_words = {
        "analyst",
        "architect",
        "consultant",
        "designer",
        "developer",
        "engineer",
        "manager",
        "recruiter",
        "scientist",
        "specialist",
        "student",
    }
    name_particles = {"da", "de", "del", "der", "la", "le", "van", "von"}

    lines = [line.strip() for line in text.splitlines() if line.strip()][:8]
    for raw_line in lines:
        line = re.sub(r"^#{1,6}\s*", "", raw_line).strip()
        labeled_name = re.match(r"(?:candidate|name)\s*[:\-]\s*(.+)", line, flags=re.IGNORECASE)
        if labeled_name:
            line = labeled_name.group(1).strip()
        else:
            line = line.split("|", maxsplit=1)[0].strip()

        normalized = line.strip(" *_`").lower()
        if normalized in ignored_lines or "@" in line or "://" in line or any(char.isdigit() for char in line):
            continue

        words = re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*\.?", line, flags=re.UNICODE)
        if not 1 <= len(words) <= 4 or len(" ".join(words)) < len(line.strip(" *_`")):
            continue
        if not all(
            word.rstrip(".").isupper()
            or word.rstrip(".")[0].isupper()
            or word.rstrip(".").lower() in name_particles
            for word in words
        ):
            continue
        if any(word.rstrip(".").lower() in role_words for word in words):
            continue

        return " ".join(words).strip(".")

    return ""


def parse_candidate_resume(path_or_text: str | Path) -> CandidateProfile:
    if isinstance(path_or_text, (str, Path)) and Path(path_or_text).exists():
        raw_text = extract_text_from_file(path_or_text)
    else:
        raw_text = clean_text(str(path_or_text))

    name = _extract_candidate_name(raw_text)

    email = ""
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", raw_text)
    if email_match:
        email = email_match.group(0)

    phone = ""
    phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}", raw_text)
    if phone_match:
        phone = phone_match.group(0)

    summary = _extract_heading_value(raw_text, "Summary") or _extract_heading_value(raw_text, "Profile")
    skills = _extract_list_items(_extract_section_block(raw_text, "Skills"))
    if not skills:
        skills = _extract_list_items(_extract_section_block(raw_text, "Core skills"))

    experience = _extract_list_items(_extract_section_block(raw_text, "Experience"))
    if not experience:
        experience = _extract_list_items(_extract_section_block(raw_text, "Work Experience"))

    education = _extract_list_items(_extract_section_block(raw_text, "Education"))

    return CandidateProfile(
        name=name,
        email=email,
        phone=phone,
        summary=summary,
        skills=skills,
        experience=experience,
        education=education,
        raw_text=raw_text,
    )


def _iter_files(directory: str | Path) -> Iterable[Path]:
    for path in sorted(Path(directory).glob("*")):
        if path.is_file():
            yield path
