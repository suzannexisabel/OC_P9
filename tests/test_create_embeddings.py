import numpy as np
import pandas as pd
import pytest

import src.rag.create_embeddings as create_embeddings

from src.rag.create_embeddings import (
    build_event_text,
    create_event_texts,
    split_text_into_chunks,
    create_event_chunks,
    generate_embeddings,
)


def test_build_event_text_contains_useful_fields():
    row = pd.Series(
        {
            "title": "Concert jazz",
            "description": "Un concert de jazz contemporain",
            "keywords": "jazz, musique",
            "dateRange": "Septembre 2026",
            "location_name": "Auditorium",
            "age": "Tout public",
            "accessibility_text": "handicap moteur",
        }
    )

    text = build_event_text(row)

    assert "Titre : Concert jazz" in text
    assert "Description : Un concert de jazz contemporain" in text
    assert "Mots-clés : jazz, musique" in text
    assert "Lieu : Auditorium" in text
    assert "Accessibilité : handicap moteur" in text


def test_create_event_texts_requires_uid_and_title():
    df_without_uid = pd.DataFrame(
        [
            {
                "title": "Concert",
            }
        ]
    )

    with pytest.raises(ValueError):
        create_event_texts(df_without_uid)

    df_without_title = pd.DataFrame(
        [
            {
                "uid": "1",
            }
        ]
    )

    with pytest.raises(ValueError):
        create_event_texts(df_without_title)


def test_create_event_texts_adds_embedding_text():
    df = pd.DataFrame(
        [
            {
                "uid": "1",
                "title": "Atelier peinture",
                "description": "Atelier créatif",
            }
        ]
    )

    result = create_event_texts(df)

    assert "embedding_text" in result.columns
    assert "Titre : Atelier peinture" in result.iloc[0]["embedding_text"]
    assert "Description : Atelier créatif" in result.iloc[0]["embedding_text"]


def test_split_text_into_chunks_with_overlap():
    text = "abcdefghij"

    chunks = split_text_into_chunks(
        text,
        chunk_size=6,
        chunk_overlap=2,
    )

    assert chunks == [
        "abcdef",
        "efghij",
    ]


def test_split_text_into_chunks_validation():
    assert split_text_into_chunks("") == []

    with pytest.raises(ValueError):
        split_text_into_chunks(
            "texte",
            chunk_size=0,
        )

    with pytest.raises(ValueError):
        split_text_into_chunks(
            "texte",
            chunk_size=10,
            chunk_overlap=-1,
        )

    with pytest.raises(ValueError):
        split_text_into_chunks(
            "texte",
            chunk_size=10,
            chunk_overlap=10,
        )


def test_create_event_chunks_creates_chunk_ids():
    df = pd.DataFrame(
        [
            {
                "uid": "42",
                "title": "Événement test",
                "embedding_text": "abcdefghij",
                "dateRange": "Septembre 2026",
            }
        ]
    )

    result = create_event_chunks(
        df,
        chunk_size=6,
        chunk_overlap=2,
    )

    assert len(result) == 2
    assert result.iloc[0]["chunk_id"] == "42_0"
    assert result.iloc[1]["chunk_id"] == "42_1"
    assert result.iloc[0]["chunk_index"] == 0
    assert result.iloc[1]["chunk_index"] == 1
    assert result.iloc[0]["uid"] == "42"


# tests sur generate_emeddings() sans appel reel de mistral

class FakeEmbeddingItem:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class FakeEmbeddingResponse:
    def __init__(self, data):
        self.data = data


class FakeEmbeddingsAPI:
    def create(self, model, inputs):
        return FakeEmbeddingResponse(
            [
                FakeEmbeddingItem(
                    index=i,
                    embedding=[float(i), float(i + 1)],
                )
                for i, _ in enumerate(inputs)
            ]
        )


class FakeMistral:
    def __init__(self, api_key):
        self.embeddings = FakeEmbeddingsAPI()


def test_generate_embeddings_returns_numpy_array(monkeypatch):
    monkeypatch.setattr(
        create_embeddings,
        "Mistral",
        FakeMistral,
    )

    result = generate_embeddings(
        [
            "texte 1",
            "texte 2",
        ],
        api_key="fake-key",
        batch_size=2,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32


def test_generate_embeddings_requires_texts():
    with pytest.raises(ValueError):
        generate_embeddings(
            [],
            api_key="fake-key",
        )


def test_generate_embeddings_requires_positive_batch_size():
    with pytest.raises(ValueError):
        generate_embeddings(
            ["texte"],
            api_key="fake-key",
            batch_size=0,
        )


