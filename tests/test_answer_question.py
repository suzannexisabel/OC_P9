import pandas as pd

import src.rag.rag_chain as rag_chain


class FakeResponse:
    content = "Voici une réponse de test."


class FakeChain:
    def invoke(self, data):
        return FakeResponse()


class FakePrompt:
    def __or__(self, llm):
        return FakeChain()


def test_answer_question_returns_expected_structure(monkeypatch):
    fake_results = pd.DataFrame(
        [
            {
                "uid": "1",
                "title": "Concert Jazz",
                "description": "Concert",
                "dateRange": "Septembre 2026",
                "location_name": "Toulouse",
                "score": 0.95,
            }
        ]
    )

    monkeypatch.setattr(
        rag_chain,
        "retrieve_events",
        lambda question, k=5: fake_results,
    )

    monkeypatch.setattr(
        rag_chain,
        "create_llm",
        lambda: object(),
    )

    monkeypatch.setattr(
        rag_chain,
        "RAG_PROMPT",
        FakePrompt(),
    )

    result = rag_chain.answer_question(
        "Je cherche du jazz"
    )

    assert result["question"] == "Je cherche du jazz"
    assert result["answer"] == "Voici une réponse de test."
    assert isinstance(result["events"], list)
    assert isinstance(result["retrieved_contexts"], list)
    assert len(result["events"]) == 1