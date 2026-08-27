import numpy as np
import pandas as pd

import src.rag.rag_chain as rag_chain


class FakeEmbeddingResponse:
    class Data:
        embedding = [0.1, 0.2, 0.3]

    data = [Data()]


class FakeEmbeddings:
    def create(self, model, inputs):
        return FakeEmbeddingResponse()


class FakeMistral:
    def __init__(self, api_key=None):
        self.embeddings = FakeEmbeddings()


class FakeIndex:
    def search(self, query_embedding, k):
        scores = np.array(
            [[0.95, 0.90, 0.85]]
        )

        indices = np.array(
            [[0, 1, 2]]
        )

        return scores, indices


def test_retrieve_events_returns_unique_events(monkeypatch):
    documents = pd.DataFrame(
        [
            {
                "uid": "1",
                "title": "Concert Jazz",
            },
            {
                "uid": "1",
                "title": "Concert Jazz - chunk 2",
            },
            {
                "uid": "2",
                "title": "Exposition",
            },
        ]
    )

    monkeypatch.setattr(
        rag_chain,
        "load_retrieval_data",
        lambda: (
            FakeIndex(),
            documents,
        ),
    )

    monkeypatch.setattr(
        rag_chain,
        "Mistral",
        FakeMistral,
    )

    monkeypatch.setattr(
        rag_chain.faiss,
        "normalize_L2",
        lambda x: None,
    )

    results = rag_chain.retrieve_events(
        "Je cherche du jazz",
        k=2,
    )

    assert len(results) == 2
    assert results["uid"].is_unique
    assert "score" in results.columns