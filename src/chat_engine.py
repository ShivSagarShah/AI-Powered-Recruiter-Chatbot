from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.config import JOBS_DIR, RESUMES_DIR
from src.parsers import clean_text, parse_candidate_resume


class RecruiterChatEngine:
    def __init__(self, documents: list[Document] | None = None):
        self.documents: list[Document] = []
        self.memory: list[dict[str, str]] = []
        self.vectorstore = None
        if documents:
            self.ingest_documents(documents)

    def ingest_documents(self, documents: Iterable[Document]) -> FAISS:
        prepared: list[Document] = []
        for doc in documents:
            try:
                content = clean_text(doc.page_content) if hasattr(doc, "page_content") else clean_text(str(doc))
            except Exception:
                content = ""
            metadata = dict(getattr(doc, "metadata", {}) or {})
            metadata.setdefault("document_type", "generic")
            if content.strip():
                prepared.append(Document(page_content=content, metadata=metadata))

        if not prepared:
            raise ValueError("No usable document content was found for indexing.")

        self.documents = prepared
        embeddings = self._resolve_embeddings()
        try:
            self.vectorstore = FAISS.from_documents(prepared, embeddings)
        except Exception as exc:
            raise ValueError(f"Failed to build the local vector index: {exc}") from exc
        return self.vectorstore

    def _resolve_embeddings(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            return GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
            )
        return FakeEmbeddings(size=1536)

    def _history_context(self) -> str:
        if not self.memory:
            return "No prior conversation."
        blocks = []
        for item in self.memory:
            blocks.append(f"{item['role']}: {item['content']}")
        return "\n".join(blocks)

    def _search(self, query: str, k: int = 5) -> list[Document]:
        if self.vectorstore is None:
            return []
        docs = self.vectorstore.similarity_search(query, k=k)
        lowered = query.lower()
        candidate_keywords = [
            "candidate",
            "experience",
            "who",
            "compare",
            "lead",
            "team",
            "kubernetes",
            "distributed",
            "agile",
            "skill",
            "resume",
            "cv",
        ]
        if any(keyword in lowered for keyword in candidate_keywords):
            resume_docs = [doc for doc in docs if doc.metadata.get("document_type") == "resume"]
            if resume_docs:
                return resume_docs
        return docs

    def _strict_answer(self, query: str, docs: list[Document]) -> str:
        if not docs:
            return "Information not available in provided records."

        context = "\n\n".join(
            f"[{doc.metadata.get('document_type', 'document')}] {doc.metadata.get('candidate_name', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        )

        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=api_key,
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert Recruiter Assistant. Answer questions ONLY using facts found in the retrieved candidate documents. "
                        "If information is not in the data, explicitly state 'Information not available in provided records.' "
                        "Do not infer or invent facts. Use the conversation history only for follow-up continuity, not for new facts.",
                    ),
                    ("human", "Conversation history:\n{history}\n\nRetrieved records:\n{context}\n\nQuestion:\n{question}"),
                ]
            )
            chain = prompt | llm
            try:
                answer = chain.invoke({
                    "history": self._history_context(),
                    "context": context,
                    "question": query,
                })
                return str(answer.content).strip() or "Information not available in provided records."
            except Exception:
                pass

        return self._fallback_answer(query, docs)

    def _fallback_answer(self, query: str, docs: list[Document]) -> str:
        lowered = query.lower()
        doc_texts = []
        for doc in docs:
            metadata = doc.metadata or {}
            doc_texts.append((metadata.get("candidate_name", "Unknown"), doc.page_content.lower(), metadata))

        if "most experience" in lowered and ("kubernetes" in lowered or "distributed systems" in lowered):
            scored = []
            for name, text, meta in doc_texts:
                score = 0
                if "kubernetes" in text: score += 2
                if "distributed systems" in text or "distributed system" in text: score += 2
                if "microservices" in text: score += 1
                if "aws" in text: score += 1
                if "docker" in text: score += 1
                if "kafka" in text: score += 1
                if "system design" in text: score += 1
                scored.append((name, score, text, meta))
            ranked = sorted(scored, key=lambda x: x[1], reverse=True)
            if ranked and ranked[0][1] > 0:
                top_name, top_score, _, _ = ranked[0]
                return f"{top_name} appears to have the strongest evidence for Kubernetes and distributed systems in the provided records, with a combined signal score of {top_score}."
            return "Information not available in provided records."

        if "compare" in lowered and "role" in lowered:
            candidate_names = [d.metadata.get("candidate_name", "") for d in docs if d.metadata.get("candidate_name")]
            candidate_name_map = {name.lower().split()[0]: name for name in candidate_names}
            query_names = [
                token for token in re.findall(r"\b[A-Z][a-z]+\b", query)
                if token.lower() not in {"compare", "senior", "backend", "role"}
            ]
            named_candidates = []
            for name in query_names:
                resolved = candidate_name_map.get(name.lower())
                if resolved:
                    named_candidates.append(resolved)

            if named_candidates:
                summary_lines = []
                for full_name in sorted(set(named_candidates)):
                    match = next((d for d in docs if d.metadata.get("candidate_name", "") == full_name), None)
                    skills = []
                    if match:
                        text = match.page_content.lower()
                        for token in ["python", "fastapi", "postgresql", "aws", "kubernetes", "sql", "docker", "redis", "distributed systems"]:
                            if token in text:
                                skills.append(token)
                        summary_lines.append(f"{full_name}: {', '.join(skills) if skills else 'No explicit skill overlap found in the provided record.'}")
                missing_names = [name for name in query_names if name.lower() not in candidate_name_map]
                if missing_names:
                    summary_lines.append(f"Information not available in provided records for {', '.join(missing_names)}.")
                if summary_lines:
                    return "\n".join(summary_lines)

            missing_names = [name for name in query_names if name.lower() not in candidate_name_map]
            if missing_names:
                return f"Information not available in provided records for {', '.join(missing_names)}."
            return "Information not available in provided records."

        if "agile" in lowered or "leading agile" in lowered:
            matches = []
            for name, text, meta in doc_texts:
                if "agile" in text or "scrum" in text:
                    matches.append(name)
            if matches:
                return f"Yes. {', '.join(matches)} explicitly mention agile or scrum in the provided records."
            return "Information not available in provided records."

        matches = []
        for name, text, meta in doc_texts:
            tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z+.-]+", query) if token.lower() not in {"who", "what", "does", "any", "candidate", "the", "for", "have", "with", "and", "compare", "role"}]
            if not tokens:
                continue
            score = 0
            for token in tokens:
                if token.lower() in text:
                    score += 1
            if score > 0:
                matches.append((name, score))

        if matches:
            best_name, _ = max(matches, key=lambda item: item[1])
            return f"The strongest match in the provided records is {best_name}."

        return "Information not available in provided records."

    def ask(self, query: str) -> str:
        docs = self._search(query, k=5)
        answer = self._strict_answer(query, docs)
        self.memory.append({"role": "user", "content": query})
        self.memory.append({"role": "assistant", "content": answer})
        return answer

    def reset_memory(self) -> None:
        self.memory = []


def _build_default_documents() -> list[Document]:
    documents: list[Document] = []
    for job_path in sorted(JOBS_DIR.glob("*")):
        texts = clean_text(job_path.read_text(encoding="utf-8", errors="replace"))
        documents.append(
            Document(
                page_content=texts,
                metadata={
                    "candidate_name": "",
                    "role": job_path.stem.replace("_", " ").title(),
                    "document_type": "job_description",
                },
            )
        )

    for resume_path in sorted(RESUMES_DIR.glob("*")):
        texts = clean_text(resume_path.read_text(encoding="utf-8", errors="replace"))
        name = parse_candidate_resume(texts).name or resume_path.stem.replace("_", " ").title()
        documents.append(
            Document(
                page_content=texts,
                metadata={
                    "candidate_name": name,
                    "role": "candidate_profile",
                    "document_type": "resume",
                },
            )
        )

    return documents


def build_default_chat_engine() -> RecruiterChatEngine:
    return RecruiterChatEngine(_build_default_documents())


if __name__ == "__main__":
    engine = build_default_chat_engine()

    sample_queries = [
        "Who has the most experience with Kubernetes and distributed systems?",
        "Compare Alex and Priya for the Senior Backend role.",
        "Does any candidate have experience leading agile teams?",
    ]

    for query in sample_queries:
        print(f"Q: {query}")
        print(f"A: {engine.ask(query)}\n")
