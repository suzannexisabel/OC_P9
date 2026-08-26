"""Évaluation du système RAG avec Ragas."""
import os
import asyncio
from pathlib import Path

import json
import pandas as pd
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
)
from ragas.dataset_schema import SingleTurnSample

from src.rag.rag_chain import answer_question

load_dotenv()

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = (RESULTS_DIR / "ragas_results.csv")

TEST_CASES_PATH = Path("evaluation/test_cases.json")

def load_questions():
    """Charge les questions du jeu de test annoté."""

    with open(
        TEST_CASES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        test_cases = json.load(file)

    return [
        test_case["question"]
        for test_case in test_cases
    ]


QUESTIONS = load_questions()

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


async def answer_question_with_retry(
    question: str,
    max_attempts: int = 3,
):
    """Appelle le RAG avec retry en cas de rate limit Mistral."""

    for attempt in range(1, max_attempts + 1):

        try:
            return answer_question(question)

        except Exception as error:

            print(
                f"\nErreur génération RAG "
                f"(tentative {attempt}/{max_attempts})"
            )

            print(
                f"{type(error).__name__}: {error}"
            )

            if attempt == max_attempts:
                print(
                    "Impossible de générer une réponse "
                    "pour cette question."
                )
                return None

            wait_time = 30 * attempt

            print(
                f"Attente de {wait_time} secondes "
                "avant nouvelle tentative..."
            )

            await asyncio.sleep(wait_time)


async def evaluate_metric_with_retry(
    metric,
    sample,
    metric_name: str,
    max_attempts: int = 2,
):
    """Évalue une métrique Ragas retry en cas d'erreur."""

    for attempt in range(1, max_attempts + 1):

        try:
            return await metric.single_turn_ascore(
                sample
            )

        except Exception as error:

            print(
                f"\nErreur {metric_name} "
                f"(tentative {attempt}/{max_attempts})"
            )
            print(
                f"{type(error).__name__}: {error}"
            )

            if attempt == max_attempts:
                print(
                    f"Impossible de calculer "
                    f"{metric_name} pour cette question."
                )
                return None

            wait_time = 20 * attempt

            print(
                f"Attente de {wait_time} secondes "
                "avant nouvelle tentative..."
            )

            await asyncio.sleep(wait_time)


async def evaluate_question(
    question: str,
    faithfulness_metric,
    context_precision_metric,
) -> dict:
    """Évalue une question avec les métriques Ragas."""

    result = await answer_question_with_retry(
        question
    )

    if result is None:
        return {
            "question": question,
            "answer": None,
            "faithfulness": None,
            "context_precision": None,
        }

    sample = SingleTurnSample(
        user_input=result["question"],
        response=result["answer"],
        retrieved_contexts=result["retrieved_contexts"],
    )

    faithfulness = await evaluate_metric_with_retry(
        faithfulness_metric,
        sample,
        "Faithfulness",
    )

    context_precision = await evaluate_metric_with_retry(
        context_precision_metric,
        sample,
        "Context Precision",
    )

    return {
        "question": question,
        "answer": result["answer"],
        "faithfulness": faithfulness,
        "context_precision": context_precision,
    }

async def main():
    """Évalue le RAG sur plusieurs scénarios."""

    evaluator_llm = create_evaluator_llm()

    faithfulness_metric = Faithfulness(
        llm=evaluator_llm
    )

    context_precision_metric = (
        LLMContextPrecisionWithoutReference(
            llm=evaluator_llm
        )
    )

    evaluation_results = []

    for number, question in enumerate(
        QUESTIONS,
        start=1,
    ):
        print(
            f"\n{'=' * 60}\n"
            f"QUESTION {number}/{len(QUESTIONS)}\n"
            f"{question}"
        )

        result = await evaluate_question(
            question,
            faithfulness_metric,
            context_precision_metric,
        )

        evaluation_results.append(result)

        # sauvegarde progressive 
        pd.DataFrame(
            evaluation_results
        ).to_csv(
            RESULTS_PATH,
            index=False,
        )

        print("\nRÉPONSE")
        print(result["answer"])

        print("\nFAITHFULNESS")
        print(result["faithfulness"])

        print("\nCONTEXT PRECISION")
        print(result["context_precision"])

        # Pause entre les deux tentatives
        if number < len(QUESTIONS):
            print(
                "\nPause de 20 secondes "
                "avant la prochaine question..."
            )

            await asyncio.sleep(20)

    df = pd.DataFrame(evaluation_results)

    print("\n" + "=" * 60)
    print("RÉSULTATS GLOBAUX")
    print("=" * 60)

    print(
        df[
            [
                "question",
                "faithfulness",
                "context_precision",
            ]
        ].to_string(index=False)
    )

    print("\nMOYENNES")

    print(
        f"Faithfulness : "
        f"{df['faithfulness'].mean():.3f}"
    )

    print(
        f"Context Precision : "
        f"{df['context_precision'].mean():.3f}"
    )

    print("\nNOMBRE D'ÉVALUATIONS VALIDES")

    print(
        f"Faithfulness : "
        f"{df['faithfulness'].notna().sum()}"
        f"/{len(df)}"
    )

    print(
        f"Context Precision : "
        f"{df['context_precision'].notna().sum()}"
        f"/{len(df)}"
    )   

if __name__ == "__main__":
    asyncio.run(main())