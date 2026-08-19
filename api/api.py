"""API FastAPI exposant le système RAG."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.rag_chain import (
    answer_question,
    load_retrieval_data
)

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