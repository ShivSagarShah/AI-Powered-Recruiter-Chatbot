from __future__ import annotations

from src.chat_engine import build_default_chat_engine


def main() -> None:
    engine = build_default_chat_engine()

    queries = [
        "Who has the most experience with Kubernetes and distributed systems?",
        "Compare Alex and Priya for the Senior Backend role.",
        "Does any candidate have experience leading agile teams?",
    ]

    answers = []
    for query in queries:
        answer = engine.ask(query)
        assert isinstance(answer, str)
        assert answer.strip(), "Answer should not be empty."
        if query == queries[0]:
            assert "strongest evidence" in answer.lower(), f"Ranking answer is not using the grounded evidence format: {answer}"
            assert any(name in answer for name in ["Amelia", "Martin"]), f"Expected a candidate name in answer: {answer}"
        elif query == queries[1]:
            assert "Priya" in answer or "Information not available in provided records" in answer, f"Unexpected answer for '{query}': {answer}"
            assert "Alex" in answer or "Information not available in provided records" in answer, f"Expected missing name handling for Alex: {answer}"
        else:
            assert "Information not available in provided records" in answer, f"Unexpected answer for '{query}': {answer}"
        answers.append(answer)

    print("Chat validation passed: grounded recruiter answers returned for sample queries.")
    for query, answer in zip(queries, answers):
        print(f"Q: {query}\nA: {answer}\n")


if __name__ == "__main__":
    main()
