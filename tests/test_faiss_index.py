import numpy as np
import pandas as pd
import pytest

from src.rag.faiss_index import (
    load_vector_data,
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
    run_faiss_indexing,
)


def test_load_vector_data(tmp_path):
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    documents = pd.DataFrame(
        [
            {"uid": "1", "title": "Événement 1"},
            {"uid": "2", "title": "Événement 2"},
        ]
    )

    np.save(
        tmp_path / "event_embeddings.npy",
        embeddings,
    )

    documents.to_parquet(
        tmp_path / "event_documents.parquet",
        index=False,
    )

    loaded_embeddings, loaded_documents = load_vector_data(
        tmp_path
    )

    assert loaded_embeddings.shape == (2, 2)
    assert loaded_embeddings.dtype == np.float32
    assert len(loaded_documents) == 2
    assert loaded_documents.iloc[0]["uid"] == "1"


def test_load_vector_data_missing_embeddings(tmp_path):
    documents = pd.DataFrame(
        [
            {"uid": "1", "title": "Événement"}
        ]
    )

    documents.to_parquet(
        tmp_path / "event_documents.parquet",
        index=False,
    )

    with pytest.raises(FileNotFoundError):
        load_vector_data(tmp_path)


def test_load_vector_data_length_mismatch(tmp_path):
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    documents = pd.DataFrame(
        [
            {"uid": "1", "title": "Événement"}
        ]
    )

    np.save(
        tmp_path / "event_embeddings.npy",
        embeddings,
    )

    documents.to_parquet(
        tmp_path / "event_documents.parquet",
        index=False,
    )

    with pytest.raises(ValueError):
        load_vector_data(tmp_path)


def test_build_faiss_index():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = build_faiss_index(embeddings)

    assert index.ntotal == 3
    assert index.d == 2


def test_build_faiss_index_requires_2d():
    embeddings = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        build_faiss_index(embeddings)


def test_build_faiss_index_rejects_empty_matrix():
    embeddings = np.empty(
        (0, 2),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        build_faiss_index(embeddings)


def test_save_and_load_faiss_index(tmp_path):
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = build_faiss_index(embeddings)

    index_path = tmp_path / "events.faiss"

    save_faiss_index(
        index,
        index_path,
    )

    assert index_path.exists()

    loaded_index = load_faiss_index(
        index_path
    )

    assert loaded_index.ntotal == 2
    assert loaded_index.d == 2


def test_load_faiss_index_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_faiss_index(
            tmp_path / "missing.faiss"
        )


def test_run_faiss_indexing(tmp_path):
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    documents = pd.DataFrame(
        [
            {"uid": "1", "title": "Événement 1"},
            {"uid": "2", "title": "Événement 2"},
        ]
    )

    np.save(
        tmp_path / "event_embeddings.npy",
        embeddings,
    )

    documents.to_parquet(
        tmp_path / "event_documents.parquet",
        index=False,
    )

    index = run_faiss_indexing(
        tmp_path
    )

    assert index.ntotal == 2
    assert (
        tmp_path / "events.faiss"
    ).exists()