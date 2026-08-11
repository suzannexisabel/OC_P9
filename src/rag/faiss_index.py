"""Création et chargement de l'index FAISS des événements."""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd


DEFAULT_VECTOR_STORE_DIR = Path("data/vector_store")


def load_vector_data(
    vector_store_dir: str | Path = DEFAULT_VECTOR_STORE_DIR,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Charge les embeddings et les métadonnées associées."""

    vector_store_path = Path(vector_store_dir)

    embeddings_path = vector_store_path / "event_embeddings.npy"
    documents_path = vector_store_path / "event_documents.parquet"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Fichier d'embeddings introuvable : {embeddings_path}"
        )

    if not documents_path.exists():
        raise FileNotFoundError(
            f"Fichier de documents introuvable : {documents_path}"
        )

    embeddings = np.load(embeddings_path).astype(np.float32)
    documents = pd.read_parquet(documents_path)

    if len(embeddings) != len(documents):
        raise ValueError(
            "Le nombre d'embeddings ne correspond pas "
            "au nombre de documents."
        )

    return embeddings, documents


def build_faiss_index(
    embeddings: np.ndarray,
) -> faiss.Index:
    """Construit un index FAISS basé sur la similarité cosinus."""

    if embeddings.ndim != 2:
        raise ValueError(
            "Les embeddings doivent être une matrice 2D."
        )

    if len(embeddings) == 0:
        raise ValueError(
            "Aucun embedding à indexer."
        )

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )

    # Normalisation nécessaire pour utiliser
    # le produit scalaire comme similarité cosinus.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    if index.ntotal != len(embeddings):
        raise RuntimeError(
            "Tous les embeddings n'ont pas été indexés."
        )

    return index


def save_faiss_index(
    index: faiss.Index,
    output_path: str | Path,
) -> None:
    """Sauvegarde l'index FAISS."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(output_path),
    )


def load_faiss_index(
    index_path: str | Path,
) -> faiss.Index:
    """Recharge un index FAISS sauvegardé."""

    index_path = Path(index_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable : {index_path}"
        )

    return faiss.read_index(str(index_path))


def run_faiss_indexing(
    vector_store_dir: str | Path = DEFAULT_VECTOR_STORE_DIR,
) -> faiss.Index:
    """Construit et sauvegarde l'index FAISS."""

    vector_store_path = Path(vector_store_dir)

    embeddings, documents = load_vector_data(
        vector_store_path
    )

    index = build_faiss_index(embeddings)

    index_path = vector_store_path / "events.faiss"

    save_faiss_index(
        index=index,
        output_path=index_path,
    )

    print(f"Nombre de chunks : {len(documents)}")
    print(f"Dimension : {embeddings.shape[1]}")
    print(f"Vecteurs indexés : {index.ntotal}")
    print(f"Index sauvegardé : {index_path}")

    return index


if __name__ == "__main__":
    run_faiss_indexing()