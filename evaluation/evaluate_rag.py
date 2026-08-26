"""Évaluation du système RAG avec Ragas."""
import os
import asyncio

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness
from ragas.dataset_schema import SingleTurnSample

from src.rag.rag_chain import answer_question

load_dotenv()

QUESTION = "Tu peux me proposer des activites creatives en septembre 2025 ?"


def create_evaluator_llm():
    """Crée le LLM utilisé par Ragas pour évaluer les réponses."""

    api_key = os.getenv("MISTRAL_KEY")

    if not api_key:
        raise EnvironmentError(
            "La variable d'environnement MISTRAL_KEY est absente."
        )

    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0,
    )

    return LangchainLLMWrapper(llm)

async def evaluate_faithfulness(result: dict):
    """Évalue la fidélité de la réponse aux contextes récupérés."""

    evaluator_llm = create_evaluator_llm()

    metric = Faithfulness(
        llm=evaluator_llm
    )

    sample = SingleTurnSample(
        user_input=result["question"],
        response=result["answer"],
        retrieved_contexts=result["retrieved_contexts"],
    )

    score = await metric.single_turn_ascore(
        sample
    )

    return score


async def main():
    """Teste le RAG et calcule le score de faithfulness."""

    result = answer_question(
        QUESTION
    )

    print("\nQUESTION")
    print(result["question"])

    print("\nRÉPONSE")
    print(result["answer"])

    print("\nNOMBRE DE CONTEXTES")
    print(
        len(result["retrieved_contexts"])
    )

    faithfulness_result = await evaluate_faithfulness(
        result
    )

    print("\nFAITHFULNESS")
    print(
        faithfulness_result
    )

if __name__ == "__main__":
    asyncio.run(main())
