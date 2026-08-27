"""API FastAPI exposant le système RAG."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.rag_chain import (
    answer_question,
    load_retrieval_data,
    reload_retrieval_data
)

from src.rag.rebuild_vector_store import rebuild_vector_store

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les ressources du RAG au démarrage de l'API."""

    load_retrieval_data()

    yield


app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API de recommandation d'événements culturels "
        "à Toulouse utilisant un système RAG."
    ),
    version="1.0.0",
    lifespan=lifespan
)


class AskRequest(BaseModel):
    """Données attendues par l'endpoint /ask."""

    question: str = Field(
        ...,
        min_length=1,
        description="Question posée au système RAG.",
        examples=["Je cherche une activité pour enfants"],
    )

    k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Nombre maximum d'événements à récupérer.",
    )


@app.get("/")
def root():
    """Vérifie que l'API est accessible."""

    return {
        "message": "API Puls-Events opérationnelle"
    }


@app.get("/health")
def health():
    """Vérifie l'état de l'API."""

    return {
        "status": "ok"
    }

@app.get("/metadata")
def metadata():
    """Retourne les informations principales du système RAG."""

    return {
        "llm": "mistral-small-latest",
        "embedding_model": "mistral-embed",
        "vector_store": "FAISS",
        "scope": "Événements et activités à Toulouse",
        "default_k": 5,
    }


@app.post("/ask")
def ask(request: AskRequest):
    """
    Pose une question au système RAG.

    La question est vectorisée puis comparée aux événements
    présents dans FAISS. Les événements les plus pertinents
    sont ensuite transmis au LLM pour générer une réponse.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide.",
        )

    try:
        result = answer_question(
            question,
            k=request.k,
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de la génération de la réponse.",
        ) from exc

@app.post("/rebuild")
def rebuild():
    """
    Reconstruit les données, les embeddings et l'index FAISS.

    Cette opération récupère à nouveau les événements OpenAgenda,
    nettoie les données, régénère les embeddings Mistral puis
    reconstruit l'index FAISS.
    """

    try:
        result = rebuild_vector_store()

        reload_retrieval_data()

        return {
            "status": "success",
            "message": "Base vectorielle reconstruite avec succès.",
            "details": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la reconstruction : {exc}",
        ) from exc