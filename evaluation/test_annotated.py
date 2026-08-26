"""Évaluation manuelle du jeu de test annoté."""

import json
from pathlib import Path

import pandas as pd

from src.rag.rag_chain import answer_question


TEST_CASES_PATH = Path("evaluation/test_cases.json")

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = RESULTS_DIR / "annotated_results.csv"


with open(
    TEST_CASES_PATH,
    "r",
    encoding="utf-8",
) as file:
    test_cases = json.load(file)


results = []


for i, test_case in enumerate(test_cases, start=1):

    question = test_case["question"]
    expected_answer = test_case["expected_answer"]

    rag_result = answer_question(question)
    generated_answer = rag_result["answer"]

    print("\n" + "=" * 60)
    print(f"QUESTION {i}/{len(test_cases)}")
    print(question)

    print("\nRÉPONSE ATTENDUE")
    print(expected_answer)

    print("\nRÉPONSE DU RAG")
    print(generated_answer)

    while True:

        label = input(
            "\nÉvaluation "
            "(c = correcte, "
            "p = partiellement correcte, "
            "i = incorrecte) : "
        ).strip().lower()

        labels = {
            "c": "correcte",
            "p": "partiellement correcte",
            "i": "incorrecte",
        }

        if label in labels:
            human_label = labels[label]
            break

        print("Choix invalide.")

    results.append(
        {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "human_label": human_label,
        }
    )

    pd.DataFrame(results).to_csv(
        RESULTS_PATH,
        index=False,
    )


print(
    "\nÉvaluation terminée. "
    "Résultats enregistrés dans "
    "evaluation/results/annotated_results.csv"
)