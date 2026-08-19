"""Tests de l'API FastAPI avec TestClient."""

import pytest

from fastapi.testclient import TestClient

from api.api import app


@pytest.fixture
def client():
    """Crée un client de test avec le lifespan FastAPI."""

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask(client):
    payload = {
        "question": "Je cherche une activité pour enfants",
        "k": 5,
    }

    response = client.post(
        "/ask",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "question" in data
    assert "answer" in data
    assert "events" in data

    assert data["question"] == payload["question"]
    assert isinstance(data["answer"], str)
    assert data["answer"].strip()

    assert isinstance(data["events"], list)
    assert len(data["events"]) <= payload["k"]


def test_empty_question(client):
    response = client.post(
        "/ask",
        json={
            "question": "",
            "k": 5,
        },
    )

    assert response.status_code == 422


def test_invalid_k(client):
    response = client.post(
        "/ask",
        json={
            "question": "Je cherche un concert",
            "k": 50,
        },
    )

    assert response.status_code == 422
