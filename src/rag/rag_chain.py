"""Chaîne RAG pour la recommandation d'événements."""

import os
from dotenv import load_dotenv
from pathlib import Path
import json

import faiss
import numpy as np
import pandas as pd

from mistralai.client import Mistral

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

VECTOR_STORE_DIR = Path("data/vector_store")

INDEX_PATH = VECTOR_STORE_DIR / "events.faiss"
DOCUMENTS_PATH = VECTOR_STORE_DIR / "event_documents.parquet"

EMBEDDING_MODEL = "mistral-embed"
LLM_MODEL = "mistral-small-latest"

api_key=os.getenv("MISTRAL_KEY")


def test_llm():
    """Teste la connexion au modèle Mistral via LangChain."""

    llm = ChatMistralAI(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0.2,
    )

    response = llm.invoke(
        "Réponds uniquement par : connexion Mistral réussie"
    )

    print(response.content)



def load_retrieval_data():
    """Charge l'index FAISS et les documents associés."""

    index = faiss.read_index(str(INDEX_PATH))
    documents = pd.read_parquet(DOCUMENTS_PATH)

    if index.ntotal != len(documents):
        raise ValueError(
            "Le nombre de vecteurs FAISS ne correspond pas "
            "au nombre de documents."
        )

    if index.ntotal == 0:
        raise ValueError("L'index FAISS est vide.")

    return index, documents


def retrieve_events(
    question: str,
    *,
    k: int = 5,
) -> pd.DataFrame:
    """Recherche les événements les plus pertinents dans FAISS."""

    index, documents = load_retrieval_data()

    embedding_client = Mistral(
        api_key=api_key
    )

    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=[question],
    )

    query_embedding = np.array(
        [response.data[0].embedding],
        dtype=np.float32,
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        k=k * 3,
    )

    results = documents.iloc[indices[0]].copy()
    results["score"] = scores[0]

    results = (
        results
        .drop_duplicates(subset="uid", keep="first")
        .head(k)
        .reset_index(drop=True)
    )

    return results


def build_context(results: pd.DataFrame) -> str:
    """Construit le contexte transmis au LLM à partir des événements récupérés."""

    context_parts = []

    for _, row in results.iterrows():

        fields = [
            ("Titre", row.get("title")),
            ("Description", row.get("description")),
            ("Description détaillée", row.get("longDescription")),
            ("Mots-clés", row.get("keywords")),

            ("Date", row.get("dateRange")),
            ("Horaires", row.get("timings")),
            ("Premier horaire", row.get("firstTiming")),
            ("Dernier horaire", row.get("lastTiming")),
            ("Prochain horaire", row.get("nextTiming")),

            ("Lieu", row.get("location_name")),
            ("Adresse", row.get("location_address")),
            ("Ville", row.get("location_city")),
            ("Code postal", row.get("location_postal_code")),

            ("Public / âge", row.get("age")),
            ("Accessibilité", row.get("accessibility_text")),
            ("Conditions", row.get("conditions")),
            ("Mode de participation", row.get("mode_participation")),

            ("Inscription", row.get("registration")),
            ("Billetterie", row.get("billetterie")),
            ("Lien", row.get("links")),

            ("Statut", row.get("status")),
        ]

        lines = []

        for label, value in fields:
            if pd.notna(value) and str(value).strip():
                lines.append(
                    f"{label} : {value}"
                )

        context_parts.append(
            "\n".join(lines)
        )

    return "\n\n---\n\n".join(context_parts)



RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Tu es un assistant spécialisé dans les événements culturels à Toulouse.

Tu dois répondre uniquement à partir des événements fournis dans le contexte.

Règles :
- N'invente jamais un événement.
- N'invente jamais une date, un lieu, un tarif ou une condition.
- Propose uniquement des événements pertinents pour la demande.
- Explique brièvement pourquoi chaque recommandation peut convenir.
- Utilise les informations de date, public, accessibilité, conditions et inscription lorsqu'elles sont disponibles.
- Si aucune information n'est disponible pour un champ, ne l'invente pas.
- Si aucun événement du contexte ne répond correctement à la demande, indique-le clairement.
- Réponds en français.
"""
        ),
        (
            "human",
            """
Question de l'utilisateur :

{question}

Événements disponibles :

{context}
"""
        ),
    ]
)


def create_llm() -> ChatMistralAI:
    """Initialise le modèle Mistral utilisé pour générer la réponse."""

    if not api_key:
        raise EnvironmentError(
            "La variable d'environnement MISTRAL_KEY est absente."
        )

    return ChatMistralAI(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0.2,
    )


def answer_question(
    question: str,
    *,
    k: int = 5,
) -> dict:
    """Génère une réponse RAG à partir des événements FAISS."""

    results = retrieve_events(
        question,
        k=k,
    )

    context = build_context(results)

    llm = create_llm()

    chain = RAG_PROMPT | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    events = json.loads(
        results.to_json(
            orient="records",
            date_format="iso",
            )
    )

    return {
        "question": question,
        "answer": response.content,
        "events": events,
    }




if __name__ == "__main__":

    result = answer_question(
        "Tu peux me proposer des activités pour enfants ?"
    )

    print("\nQUESTION\n")
    print(result["question"])

    print("\nRÉPONSE\n")
    print(result["answer"])

    print("\nÉVÉNEMENTS RETROUVÉS\n")

    for event in result["events"]:
        print(
            f"- {event.get('title')} | "
            f"{event.get('dateRange')} | "
            f"{event.get('location_name')} | "
            f"score={event.get('score')}"
        )