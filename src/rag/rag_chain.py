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
_index = None
_documents = None


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
    """Charge l'index FAISS et les documents une seule fois en mémoire."""

    global _index, _documents

    if _index is not None and _documents is not None:
        return _index, _documents

    print("Chargement de l'index FAISS et des documents...")

    index = faiss.read_index(str(INDEX_PATH))
    documents = pd.read_parquet(DOCUMENTS_PATH)

    if index.ntotal != len(documents):
        raise ValueError(
            "Le nombre de vecteurs FAISS ne correspond pas "
            "au nombre de documents."
        )

    if index.ntotal == 0:
        raise ValueError("L'index FAISS est vide.")

    _index = index
    _documents = documents

    print(
        f"Base vectorielle chargée : "
        f"{_index.ntotal} vecteurs."
    )

    return _index, _documents


def reload_retrieval_data():
    """Recharge l'index FAISS et les documents depuis le disque."""

    global _index, _documents

    _index = None
    _documents = None

    return load_retrieval_data()


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


def build_context_list(results: pd.DataFrame) -> list[str]:
    """Construit une liste de contextes, un texte par événement."""

    contexts = []

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

        contexts.append(
            "\n".join(lines)
        )

    return contexts



RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Tu es un assistant spécialisé dans les événements culturels à Toulouse.

Tu dois répondre uniquement à partir des événements fournis dans le contexte.

Règles :
- N'invente jamais un événement.
- N'invente jamais une date, un lieu, un tarif, une condition ou une information absente du contexte.
- Ne recommande un événement que s'il correspond clairement aux critères exprimés dans la question.
- Il n'est pas nécessaire de recommander tous les événements présents dans le contexte.
- Privilégie les événements les plus pertinents pour la demande de l'utilisateur.
- Explique brièvement pourquoi chaque recommandation correspond à la demande.
- Utilise les informations de date, lieu, public, accessibilité, conditions ou inscription uniquement lorsqu'elles sont utiles à la demande.
- Si une information n'est pas disponible dans le contexte, ne l'invente pas.
- Si aucun événement du contexte ne répond correctement à la demande, indique-le clairement.
- Fais une réponse concise, naturelle et facile à lire.
- Évite de répéter les mêmes informations dans un résumé ou un tableau final.
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
    retrieved_contexts = build_context_list(results)

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
        "retrieved_contexts": retrieved_contexts,
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
            f"\n- {event.get('title')} | "
            f"{event.get('dateRange')} | "
            f"{event.get('location_name')} | "
            f"score={event.get('score')}"
        )

    print("\nCONTEXTES RAGAS\n")

    for context in result["retrieved_contexts"]:
        print(context)
        print("\n---\n")