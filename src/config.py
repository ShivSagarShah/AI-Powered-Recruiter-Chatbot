from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
JOBS_DIR = DATA_DIR / "jds"
RESUMES_DIR = DATA_DIR / "resumes"


class Settings(BaseModel):
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    jobs_dir: Path = JOBS_DIR
    resumes_dir: Path = RESUMES_DIR
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")


settings = Settings()
