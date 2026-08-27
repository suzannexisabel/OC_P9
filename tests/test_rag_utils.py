import pandas as pd

from src.rag.rag_chain import (
    build_context,
    build_context_list,
)


def test_build_context_contains_useful_fields():
    data = pd.DataFrame(
        [
            {
                "title": "Concert Jazz",
                "description": "Concert de jazz contemporain",
                "longDescription": None,
                "keywords": "jazz, musique",
                "dateRange": "10 septembre 2026",
                "timings": "10/09/2026 à 20h",
                "firstTiming": "10/09/2026 à 20h",
                "lastTiming": "10/09/2026 à 22h",
                "nextTiming": "10/09/2026 à 20h",
                "location_name": "Salle de concert",
                "location_address": "1 rue Exemple",
                "location_city": "Toulouse",
                "location_postal_code": "31000",
                "age": "Tout public",
                "accessibility_text": "Accessible PMR",
                "conditions": "Gratuit",
                "mode_participation": "Présentiel",
                "registration": "Réservation obligatoire",
                "billetterie": None,
                "links": "https://example.com",
                "status": "Programmé",
            }
        ]
    )

    context = build_context(data)

    assert "Titre : Concert Jazz" in context
    assert "Date : 10 septembre 2026" in context
    assert "Lieu : Salle de concert" in context
    assert "Accessibilité : Accessible PMR" in context
    assert "Conditions : Gratuit" in context


def test_build_context_ignores_missing_values():
    data = pd.DataFrame(
        [
            {
                "title": "Exposition",
                "description": None,
                "dateRange": "Novembre 2026",
                "location_name": "Musée",
            }
        ]
    )

    context = build_context(data)

    assert "Titre : Exposition" in context
    assert "Description :" not in context
    assert "Titre : nan" not in context


def test_build_context_list_returns_one_context_per_event():
    data = pd.DataFrame(
        [
            {
                "title": "Événement 1",
                "description": "Description 1",
            },
            {
                "title": "Événement 2",
                "description": "Description 2",
            },
        ]
    )

    contexts = build_context_list(data)

    assert isinstance(contexts, list)
    assert len(contexts) == 2
    assert "Titre : Événement 1" in contexts[0]
    assert "Titre : Événement 2" in contexts[1]