"""Reconstruction complète de la base vectorielle."""

from src.data.clean_events import run_pipeline
from src.rag.create_embeddings import run_embedding_pipeline
from src.rag.faiss_index import run_faiss_indexing


def rebuild_vector_store(
        max_agendas: int | None = None,
) -> dict:
    """Reconstruit entièrement les données et l'index FAISS."""

    print("\n=== 1. Import et nettoyage des événements ===\n")

    clean_df = run_pipeline(max_agendas=max_agendas)

    print("\n=== 2. Création des embeddings ===\n")

    _, embeddings = run_embedding_pipeline()

    print("\n=== 3. Construction de l'index FAISS ===\n")

    index = run_faiss_indexing()

    print("\n=== Reconstruction terminée ===\n")

    return {
        "clean_events": len(clean_df),
        "chunks": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "indexed_vectors": int(index.ntotal),
    }


if __name__ == "__main__":
    result = rebuild_vector_store(
        max_agendas=35
    )
    print(result)