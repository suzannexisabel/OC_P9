"""Création des embeddings Mistral pour les événements OpenAgenda."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mistralai.client import Mistral

from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = "mistral-embed"
DEFAULT_BATCH_SIZE = 32


def _valid_text(value: Any) -> str | None:
    """Retourne une chaîne propre ou None si la valeur est absente."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    return text or None


def build_event_text(row: pd.Series) -> str:
    """Construit le texte sémantique à vectoriser pour un événement."""

    fields = [
        ("Titre", row.get("title")),
        ("Description", row.get("description")),
        ("Description détaillée", row.get("longDescription")),
        ("Mots-clés", row.get("keywords")),
        ("Date", row.get("dateRange")),
        ("Horaires", row.get("timings")),
        ("Lieu", row.get("location_name")),
        ("Adresse", row.get("location_address")),
        ("Conditions", row.get("conditions")),
        ("Public", row.get("age")),
        ("Accessibilité", row.get("accessibility_text")),
        ("Mode de participation", row.get("mode_participation")),
    ]

    parts: list[str] = []

    for label, value in fields:
        text = _valid_text(value)

        if text:
            parts.append(f"{label} : {text}")

    return "\n".join(parts)


def create_event_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne contenant le texte destiné aux embeddings."""
    if "uid" not in df.columns:
        raise ValueError("La colonne 'uid' est obligatoire.")

    if "title" not in df.columns:
        raise ValueError("La colonne 'title' est obligatoire.")

    result = df.copy()
    result["embedding_text"] = result.apply(build_event_text, axis=1)

    empty_documents = result["embedding_text"].str.strip().eq("")

    if empty_documents.any():
        raise ValueError(
            f"{empty_documents.sum()} événement(s) ne contiennent aucun texte."
        )

    return result



def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    """Découpe un texte en chunks avec chevauchement."""

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size doit être supérieur à zéro.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap ne peut pas être négatif.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap doit être inférieur à chunk_size."
        )

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = end - chunk_overlap

    return chunks



def create_event_chunks(
    df: pd.DataFrame,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> pd.DataFrame:
    """Transforme chaque événement en un ou plusieurs chunks."""

    if "embedding_text" not in df.columns:
        raise ValueError(
            "La colonne 'embedding_text' est obligatoire."
        )

    chunk_rows = []

    for _, row in df.iterrows():
        chunks = split_text_into_chunks(
            text=row["embedding_text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk_text in enumerate(chunks):
            chunk_rows.append(
                {
                    "uid": row["uid"],
                    "title": row["title"],
                    "chunk_id": f"{row['uid']}_{chunk_index}",
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "description": row.get("description"),
                    "longDescription": row.get("longDescription"),
                    "dateRange": row.get("dateRange"),
                    "firstTiming": row.get("firstTiming"),
                    "lastTiming": row.get("lastTiming"),
                    "location_name": row.get("location_name"),
                    "location_address": row.get("location_address"),
                    "location_postal_code": row.get(
                        "location_postal_code"
                    ),
                    "location_latitude": row.get(
                        "location_latitude"
                    ),
                    "location_longitude": row.get(
                        "location_longitude"
                    ),
                    "image": row.get("image"),
                    "registration": row.get("registration"),
                    "status": row.get("status"),
                }
            )

    return pd.DataFrame(chunk_rows)




def generate_embeddings(
    texts: list[str],
    *,
    api_key: str,
    model: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Génère les embeddings Mistral par lots."""

    if not texts:
        raise ValueError("Aucun texte à vectoriser.")

    if batch_size <= 0:
        raise ValueError("batch_size doit être strictement positif.")

    client = Mistral(api_key=api_key)
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        response = client.embeddings.create(
            model=model,
            inputs=batch,
        )

        # Sécurise l'ordre des résultats avec l'index retourné par l'API.
        ordered_data = sorted(
            response.data,
            key=lambda item: item.index,
        )

        batch_embeddings = [
            item.embedding
            for item in ordered_data
        ]

        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                "Le nombre d'embeddings reçus ne correspond pas "
                "au nombre de textes envoyés."
            )

        all_embeddings.extend(batch_embeddings)

        processed = min(start + batch_size, len(texts))
        print(f"Embeddings générés : {processed}/{len(texts)}")

    embeddings = np.asarray(all_embeddings, dtype=np.float32)

    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Le nombre final de vecteurs ne correspond pas "
            "au nombre d'événements."
        )

    return embeddings


def save_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: str | Path,
) -> None:
    """Sauvegarde les vecteurs et les documents associés."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if len(df) != len(embeddings):
        raise ValueError(
            "Le DataFrame et la matrice d'embeddings "
            "n'ont pas la même longueur."
        )

    # Matrice numérique optimisée pour FAISS.
    np.save(
        output_path / "event_embeddings.npy",
        embeddings,
    )

    # Documents et métadonnées dans le même ordre que les vecteurs.
    metadata_columns = [
        column
        for column in [
            "uid",
            "title",
            "chunk_id",
            "chunk_text",
            "description",
            "longDescription",
            "dateRange",
            "firstTiming",
            "lastTiming",
            "location_name",
            "location_address",
            "location_postal_code",
            "location_latitude",
            "location_longitude",
            "image",
            "registration",
            "status",
        ]
        if column in df.columns
    ]

    df.loc[:, metadata_columns].to_parquet(
        output_path / "event_documents.parquet",
        index=False,
    )


def run_embedding_pipeline(
    input_path: str | Path = "data/processed/events_clean.parquet",
    output_dir: str | Path = "data/vector_store",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Charge les événements, crée les textes et génère les embeddings."""

    api_key = os.getenv("MISTRAL_KEY")

    if not api_key:
        raise EnvironmentError(
            "La variable d'environnement MISTRAL_KEY est absente."
        )

    df = pd.read_parquet(input_path)
    df_documents = create_event_texts(df)

    df_chunks = create_event_chunks(
        df_documents,
        chunk_size=1000,
        chunk_overlap=150,
    )

    embeddings = generate_embeddings(
        df_chunks["chunk_text"].tolist(),
        api_key=api_key,
        batch_size=batch_size,
    )

    save_embeddings(
        df=df_chunks,
        embeddings=embeddings,
        output_dir=output_dir,
    )

    print(f"Nombre d'événements : {len(df_documents)}")
    print(f"Forme des embeddings : {embeddings.shape}")
    print(f"Fichiers sauvegardés dans : {Path(output_dir)}")

    return df_documents, embeddings


if __name__ == "__main__":
    run_embedding_pipeline()