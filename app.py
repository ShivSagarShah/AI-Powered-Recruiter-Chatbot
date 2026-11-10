from __future__ import annotations

import json
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.chat_engine import RecruiterChatEngine
from src.config import JOBS_DIR, RESUMES_DIR
from src.generators import draft_recruiter_email, generate_interview_questions
from src.matcher import rank_candidates
from src.parsers import clean_text, parse_candidate_resume, parse_job_description


st.set_page_config(page_title="Recruiter Copilot", page_icon="🤝", layout="wide")

APP_TITLE = "AI Recruiter Copilot"
FIT_COLORS = {
    "High": ("#15803d", "#dcfce7"),
    "Medium": ("#b45309", "#fef3c7"),
    "Low": ("#b91c1c", "#fee2e2"),
}


def _render_fit_badge(fit_level: str) -> None:
    color, background = FIT_COLORS.get(fit_level, ("#475569", "#e2e8f0"))
    st.markdown(
        f'<span style="display:inline-block;padding:0.25rem 0.7rem;border-radius:999px;'
        f'font-size:0.82rem;font-weight:700;color:{color};background:{background};">'
        f'{escape(fit_level)} fit</span>',
        unsafe_allow_html=True,
    )


def _render_score_bar(score: int, fit_level: str) -> None:
    color, _ = FIT_COLORS.get(fit_level, ("#475569", "#e2e8f0"))
    st.markdown(
        f'<div style="height:0.55rem;background:var(--secondary-background-color);'
        f'border-radius:999px;overflow:hidden;">'
        f'<div style="width:{score}%;height:100%;background:{color};border-radius:999px;"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_skill_chips(skills: list[str], label: str, background: str, color: str) -> None:
    visible_skills = skills[:8]

    def render_chips(items: list[str]) -> None:
        chips = []
        for skill in items:
            title = escape(skill, quote=True)
            display_skill = skill if len(skill) <= 48 else f"{skill[:45]}..."
            chips.append(
                f'<span title="{title}" style="display:inline-block;margin:0.15rem;padding:0.25rem 0.55rem;'
                f'border-radius:999px;background:{background};color:{color};">{escape(display_skill)}</span>'
            )
        st.markdown(" ".join(chips), unsafe_allow_html=True)

    render_chips(visible_skills)
    remaining_skills = skills[len(visible_skills):]
    if remaining_skills:
        with st.expander(f"Show {len(remaining_skills)} more {label}"):
            render_chips(remaining_skills)


def get_default_sample_jd_names() -> list[str]:
    try:
        return sorted(p.name for p in JOBS_DIR.glob("*"))
    except Exception:
        return []


def get_default_sample_resume_data() -> list[dict]:
    records = []
    try:
        for resume_path in sorted(RESUMES_DIR.glob("*")):
            try:
                text = clean_text(resume_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                text = ""
            records.append({
                "candidate_name": parse_candidate_resume(text).name or resume_path.stem.replace("_", " ").title(),
                "resume_text": text,
            })
    except Exception:
        return []
    return records


def read_uploaded_file(file_obj) -> str:
    if file_obj is None:
        raise ValueError("No file was provided.")

    file_name = getattr(file_obj, "name", "")
    try:
        if file_name.lower().endswith(".pdf"):
            from pypdf import PdfReader

            try:
                pdf_reader = PdfReader(file_obj)
                pages = [page.extract_text() or "" for page in pdf_reader.pages]
            except Exception as exc:  # pragma: no cover - defensive UI fallback
                raise ValueError(f"The PDF could not be read: {exc}") from exc
            text = clean_text("\n".join(pages))
            if not text.strip():
                raise ValueError(f"The PDF '{file_name}' appears to be empty or unreadable.")
            return text

        raw = file_obj.read()
        if raw is None:
            raise ValueError(f"The file '{file_name}' could not be read.")
        decoded = raw.decode("utf-8", errors="strict")
        text = clean_text(decoded)
        if not text.strip():
            raise ValueError(f"The file '{file_name}' is empty.")
        return text
    except UnicodeDecodeError as exc:
        raise ValueError(f"The file '{file_name}' is not valid UTF-8 text. Please upload a plain-text or PDF file.") from exc
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"The file '{file_name}' could not be processed. Please check that it is valid and not corrupted.") from exc


def build_documents_for_active_data(jd_text: str, resume_entries: list[dict]) -> list[dict]:
    docs = [{
        "candidate_name": "",
        "role": "selected_job_description",
        "document_type": "job_description",
        "text": jd_text,
    }]
    for entry in resume_entries:
        docs.append({
            "candidate_name": entry.get("candidate_name", "Candidate"),
            "role": "candidate_profile",
            "document_type": "resume",
            "text": entry.get("resume_text", ""),
        })
    return docs


def ensure_session_defaults() -> None:
    if "rankings" not in st.session_state:
        st.session_state.rankings = []
    if "selected_candidate" not in st.session_state:
        st.session_state.selected_candidate = None
    if "drafted_email" not in st.session_state:
        st.session_state.drafted_email = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""
    if "jd_title" not in st.session_state:
        st.session_state.jd_title = "Selected Role"
    if "active_resume_entries" not in st.session_state:
        st.session_state.active_resume_entries = get_default_sample_resume_data()


def analyze_and_index() -> None:
    jd_text = st.session_state.get("jd_text", "")
    resumes = st.session_state.get("active_resume_entries", [])
    if not jd_text or not jd_text.strip():
        st.warning("Please upload or choose a job description before analyzing.")
        return
    if not resumes:
        st.warning("Please add at least one candidate CV to analyze.")
        return

    progress_bar = st.progress(0, text="Preparing candidate analysis...")
    current_step = st.empty()
    partial_results = st.empty()
    completed_results = []

    def update_candidate_progress(index, total, candidate_name, state, evaluation) -> None:
        if state == "started":
            current_step.info(f"Analyzing {candidate_name} ({index} of {total})...")
            return

        progress_bar.progress(index / total, text=f"Processed {index} of {total} resumes")
        if state == "completed" and evaluation is not None:
            completed_results.append(evaluation)
            completed_results.sort(key=lambda item: (-item.match_score, item.candidate_name.lower()))
            preview = "\n".join(
                f"{position}. **{escape(item.candidate_name)}** — "
                f"{item.match_score}/100 ({item.fit_level})"
                for position, item in enumerate(completed_results, start=1)
            )
            partial_results.markdown("#### Completed evaluations\n" + preview)
            current_step.success(f"Finished {candidate_name} ({index} of {total})")
        else:
            current_step.warning(f"Skipped {candidate_name} ({index} of {total}); no usable evaluation was produced.")

    try:
        rankings = rank_candidates(resumes, jd_text, progress_callback=update_candidate_progress)
    except Exception as exc:
        current_step.error("Candidate analysis failed.")
        st.error(f"Ranking failed. Please check the uploaded CVs and JD: {exc}")
        return
    progress_bar.progress(1.0, text=f"Completed analysis for {len(resumes)} resumes")
    current_step.info("Candidate evaluation complete. Building recruiter assistant index...")

    if not rankings:
        current_step.error("No candidate evaluations were produced.")
        st.warning("No candidates were scored. Please verify the CV content and selected JD.")
        return

    st.session_state.rankings = rankings
    st.session_state.selected_candidate = rankings[0].candidate_name

    docs = []
    docs.append({
        "page_content": jd_text,
        "metadata": {"candidate_name": "", "role": "selected_job_description", "document_type": "job_description"},
    })
    for entry in resumes:
        resume_text = str(entry.get("resume_text", "") or "").strip()
        if not resume_text:
            continue
        docs.append({
            "page_content": resume_text,
            "metadata": {
                "candidate_name": parse_candidate_resume(resume_text).name or entry.get("candidate_name", "Candidate"),
                "role": "candidate_profile",
                "document_type": "resume",
            },
        })

    if not docs:
        current_step.error("No usable documents were available to index.")
        st.warning("No usable document content was found. Please check uploaded files and try again.")
        return

    try:
        from langchain_core.documents import Document

        engine = RecruiterChatEngine([Document(page_content=item["page_content"], metadata=item["metadata"]) for item in docs])
        st.session_state.chat_engine = engine
        st.session_state.chat_history = []
        current_step.success("Candidate evaluation and indexing complete.")
        st.success("Analysis complete. Ranking and retriever are ready.")
    except Exception as exc:
        current_step.error("Candidate evaluation finished, but recruiter assistant indexing failed.")
        st.error(f"The retriever could not be initialized. Please check the environment and uploaded inputs: {exc}")


def render_ranking_tab() -> None:
    rankings = st.session_state.get("rankings", [])
    if not rankings:
        st.info("Upload a JD and candidates, then click Analyze & Index to generate rankings.")
        return

    heuristic_results = [item for item in rankings if item.fallback_reason]
    if heuristic_results:
        candidate_names = ", ".join(item.candidate_name for item in heuristic_results)
        reasons = "; ".join(sorted({item.fallback_reason for item in heuristic_results if item.fallback_reason}))
        st.warning(
            f"AI evaluation was unavailable for {candidate_names}; heuristic scoring was used instead."
        )
        with st.expander("AI evaluation details"):
            st.caption("Reason")
            st.text(reasons)

    st.subheader("Leaderboard")
    for rank, item in enumerate(rankings, start=1):
        with st.container(border=True):
            rank_col, candidate_col, score_col, fit_col = st.columns([0.45, 2.4, 1.3, 1])
            with rank_col:
                st.markdown(f"### {rank}")
            with candidate_col:
                st.markdown(f"**{escape(item.candidate_name)}**")
                st.caption(f"{item.years_of_experience_est:g} years estimated experience")
            with score_col:
                st.markdown(f"**{item.match_score}/100**")
                _render_score_bar(item.match_score, item.fit_level)
            with fit_col:
                _render_fit_badge(item.fit_level)

    st.subheader("Candidate Breakdown")
    for item in rankings:
        with st.container(border=True):
            heading_col, badge_col = st.columns([3, 1])
            with heading_col:
                st.markdown(f"### {escape(item.candidate_name)}")
                st.markdown(escape(item.overall_summary))
            with badge_col:
                _render_fit_badge(item.fit_level)
                st.metric("Match score", f"{item.match_score}/100")
            _render_score_bar(item.match_score, item.fit_level)
            st.caption(f"Estimated experience: {item.years_of_experience_est:g} years")

            matched_col, missing_col = st.columns(2)
            with matched_col:
                st.markdown("**✅ Matched skills**")
                if item.matched_skills:
                    _render_skill_chips(item.matched_skills, "matched skills", "#dcfce7", "#166534")
                else:
                    st.caption("No strong skill matches found.")
            with missing_col:
                st.markdown("**⚠️ Skill gaps**")
                if item.missing_skills:
                    _render_skill_chips(item.missing_skills, "skill gaps", "#fee2e2", "#991b1b")
                else:
                    st.caption("No major gaps identified.")


def render_tools_tab() -> None:
    rankings = st.session_state.get("rankings", [])
    if not rankings:
        st.info("Analyze candidates first to enable interview and email tools.")
        return

    candidate_names = [item.candidate_name for item in rankings]
    selected = st.selectbox("Candidate", candidate_names, index=0)
    selected_eval = next((item for item in rankings if item.candidate_name == selected), rankings[0])
    st.session_state.selected_candidate = selected

    if st.button("Generate Interview Questions"):
        jd_text = st.session_state.get("jd_text", "")
        if not jd_text:
            st.warning("Choose a JD first.")
        else:
            qset = generate_interview_questions(selected_eval, jd_text)
            st.session_state.interview_questions = qset
            st.success("Interview questions generated.")

    if "interview_questions" in st.session_state:
        qset = st.session_state.interview_questions
        if qset.get("technical_questions"):
            st.subheader("Technical Questions")
            for idx, item in enumerate(qset["technical_questions"], start=1):
                st.markdown(f"### {idx}. {item['question']}")
                st.markdown(f"**Good answer:** {item['good_answer']}")
                st.markdown(f"**Red flag:** {item['red_flag']}")
                st.markdown("---")

        if qset.get("probing_questions"):
            st.subheader("Probing Questions")
            for idx, item in enumerate(qset["probing_questions"], start=1):
                st.markdown(f"### {idx}. {item['question']}")
                st.markdown(f"**Good answer:** {item['good_answer']}")
                st.markdown(f"**Red flag:** {item['red_flag']}")
                st.markdown("---")

    st.subheader("Email Composer")
    email_type = st.selectbox("Email Type", ["interview_invitation", "assessment_instructions", "polite_rejection"])
    custom_notes = st.text_area("Custom Notes", value=f"Candidate fit: {selected_eval.fit_level}; match score: {selected_eval.match_score}/100.")
    if st.button("Draft Email"):
        details = {"notes": custom_notes}
        if email_type == "interview_invitation":
            details.update({
                "round_format": "technical panel",
                "interviewer_name": "Hiring Manager",
                "availability_slots": ["Tue 3:00 PM PT", "Thu 10:30 AM PT"],
            })
        elif email_type == "assessment_instructions":
            details.update({
                "platform_link": "https://example.com/assessment",
                "deadline": "48 hours from now",
                "guidelines": ["Use your own work", "Keep answers concise", "Do not share the prompt"],
            })
        email_text = draft_recruiter_email(selected_eval, st.session_state.get("jd_title", "Selected Role"), email_type, details)
        st.session_state.drafted_email = email_text
        st.success("Email draft generated.")

    if st.session_state.get("drafted_email"):
        st.markdown(st.session_state.drafted_email)
        if st.button("Copy Email"):
            components.html(
                f"<script>navigator.clipboard.writeText({json.dumps(st.session_state.drafted_email)});</script>",
                height=0,
            )
            st.toast("Email copied to clipboard.")


def render_chat_tab() -> None:
    engine = st.session_state.get("chat_engine")
    if engine is None:
        st.info("Run analysis first to load documents into the recruiter assistant.")
        return

    st.subheader("Recruiter Assistant")

    chip_labels = [
        "Compare top 2 candidates",
        "Who has the best cloud background?",
        "Who has the most experience with Kubernetes and distributed systems?",
    ]
    chip_cols = st.columns(len(chip_labels))
    for col, label in zip(chip_cols, chip_labels):
        with col:
            if st.button(label, key=f"chip_{label}"):
                st.session_state.chat_history.append({"role": "user", "content": label})
                answer = engine.ask(label)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

    for entry in st.session_state.get("chat_history", []):
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    prompt = st.chat_input("Ask the recruiter assistant about the active candidates and JD")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        answer = engine.ask(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


def sidebar_setup() -> None:
    st.sidebar.header("Recruitment Inputs")
    uploaded_jd = st.sidebar.file_uploader("Upload Job Description (.pdf, .txt, .md)", type=["pdf", "txt", "md"])

    sample_jds = ["None"] + get_default_sample_jd_names()
    selected_sample = st.sidebar.selectbox("Or choose a sample JD", sample_jds, index=0)

    uploaded_resumes = st.sidebar.file_uploader(
        "Upload candidate CVs (.pdf, .txt, .md)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_resumes:
        resume_entries = []
        for file_obj in uploaded_resumes:
            text = read_uploaded_file(file_obj)
            name = parse_candidate_resume(text).name or Path(file_obj.name).stem.replace("_", " ").title()
            resume_entries.append({"candidate_name": name, "resume_text": text})
        st.session_state.active_resume_entries = resume_entries
    elif selected_sample != "None":
        jd_path = JOBS_DIR / selected_sample
        jd_text = clean_text(jd_path.read_text(encoding="utf-8", errors="replace"))
        st.session_state.jd_text = jd_text
        st.session_state.jd_title = parse_job_description(jd_text).title or selected_sample.replace(".md", "").replace("_", " ").title()
        st.session_state.active_resume_entries = get_default_sample_resume_data()
    else:
        st.session_state.active_resume_entries = get_default_sample_resume_data()

    if uploaded_jd is not None:
        st.session_state.jd_text = read_uploaded_file(uploaded_jd)
        st.session_state.jd_title = Path(uploaded_jd.name).stem.replace("_", " ").title()
    elif selected_sample != "None":
        jd_path = JOBS_DIR / selected_sample
        st.session_state.jd_text = clean_text(jd_path.read_text(encoding="utf-8", errors="replace"))
        st.session_state.jd_title = parse_job_description(st.session_state.jd_text).title or selected_sample.replace(".md", "").replace("_", " ").title()

    if st.sidebar.button("Analyze & Index"):
        analyze_and_index()


def main() -> None:
    ensure_session_defaults()
    sidebar_setup()

    st.title(APP_TITLE)
    st.caption("Grounded recruiting assistant for CV-to-JD matching, gap analysis, and recruiter workflows.")

    tabs = st.tabs(["Candidate Ranking & Gap Analysis", "Tools", "Recruiter Assistant"])

    with tabs[0]:
        render_ranking_tab()

    with tabs[1]:
        render_tools_tab()

    with tabs[2]:
        render_chat_tab()


if __name__ == "__main__":
    main()
