import pandas as pd
import pytest

from src.data.clean_events import (
    _clean_text,
    _format_age,
    _format_accessibility,
    _format_registration,
    _format_timing,
    _merge_duplicate_events,
    clean_events,
)


def test_clean_text_removes_html_and_extra_spaces():
    value = "<p>Bonjour   Toulouse</p>"

    result = _clean_text(value)

    assert result == "Bonjour Toulouse"


def test_format_age():
    assert _format_age({"min": 3, "max": 6}) == "De 3 à 6 ans"
    assert _format_age({"min": 12}) == "À partir de 12 ans"
    assert _format_age({"max": 5}) == "Jusqu'à 5 ans"
    assert _format_age(None) is None


def test_format_accessibility():
    value = {
        "mi": True,
        "vi": True,
        "hi": False,
    }

    result = _format_accessibility(value)

    assert "handicap moteur" in result
    assert "handicap visuel" in result
    assert "handicap auditif" not in result


def test_format_registration():
    value = [
        {
            "type": "email",
            "value": "contact@example.com",
        },
        {
            "type": "phone",
            "value": "0102030405",
        },
    ]

    result = _format_registration(value)

    assert "Email : contact@example.com" in result
    assert "Téléphone : 0102030405" in result


def test_format_timing():
    value = {
        "begin": "2026-09-10T18:00:00",
        "end": "2026-09-10T20:00:00",
    }

    result = _format_timing(value)

    assert "10/09/2026" in result
    assert "18h00" in result
    assert "20h00" in result


def test_merge_duplicate_events_keeps_unique_uid():
    df = pd.DataFrame(
        [
            {
                "uid": "1",
                "title": "Ancien titre",
                "description": None,
                "keywords": ["jazz"],
                "updatedAt": "2026-01-01T10:00:00Z",
                "agenda_uid": "a1",
                "agenda_title": "Agenda 1",
            },
            {
                "uid": "1",
                "title": "Nouveau titre",
                "description": "Description complète",
                "keywords": ["jazz", "concert"],
                "updatedAt": "2026-01-02T10:00:00Z",
                "agenda_uid": "a2",
                "agenda_title": "Agenda 2",
            },
        ]
    )

    result = _merge_duplicate_events(df)

    assert len(result) == 1
    assert result.iloc[0]["uid"] == "1"
    assert result.iloc[0]["title"] == "Nouveau titre"
    assert result.iloc[0]["description"] == "Description complète"
    assert set(result.iloc[0]["keywords"]) == {"jazz", "concert"}
    assert result.iloc[0]["agenda_uids"] == ["a2", "a1"]



def test_clean_events_requires_uid_and_title():
    df = pd.DataFrame(
        [
            {
                "description": "Test",
            }
        ]
    )

    with pytest.raises(ValueError):
        clean_events(df)


def test_clean_events_formats_business_fields():
    df = pd.DataFrame(
        [
            {
                "uid": "1",
                "title": "Atelier enfants",
                "description": "<p>Atelier créatif</p>",
                "age": {
                    "min": 3,
                    "max": 6,
                },
                "accessibility": {
                    "mi": True,
                },
                "attendanceMode": 1,
                "status": 1,
                "location": {
                    "name": "Musée",
                    "address": "1 rue Exemple",
                    "city": "Toulouse",
                    "postalCode": "31000",
                    "department": "haute-garonne",
                    "region": "occitania",
                },
                "registration": [
                    {
                        "type": "link",
                        "value": "https://example.com",
                    }
                ],
            },
            {
                "uid": "2",
                "title": "Concert",
                "description": "Concert test",
                "age": {
                    "min": 12,
                },
                "accessibility": {
                    "vi": True,
                },
                "attendanceMode": 2,
                "status": 2,
                "location": {
                    "name": "Salle de concert",
                    "address": "2 rue Exemple",
                    "city": "Toulouse",
                    "postalCode": "31000",
                    "department": "alto garona",
                    "region": "occitanie",
                },
                "registration": [
                    {
                        "type": "email",
                        "value": "contact@example.com",
                    }
                ],
            },
        ]
    )

    result = clean_events(df)

    row = result.loc[result["uid"] == "1"].iloc[0]

    assert row["description"] == "Atelier créatif"
    assert row["age"] == "De 3 à 6 ans"
    assert row["accessibility_text"] == "handicap moteur"
    assert row["mode_participation"] == "Présentiel"
    assert row["status"] == "Programmé"
    assert "Lien : https://example.com" in row["registration"]

def test_clean_events_removes_constant_geographic_columns():
    df = pd.DataFrame(
        [
            {
                "uid": "1",
                "title": "Événement 1",
                "location": {
                    "name": "Lieu 1",
                    "city": "Toulouse",
                    "department": "haute-garonne",
                    "region": "occitania",
                },
            },
            {
                "uid": "2",
                "title": "Événement 2",
                "location": {
                    "name": "Lieu 2",
                    "city": "Toulouse",
                    "department": "alto garona",
                    "region": "occitanie",
                },
            },
        ]
    )

    result = clean_events(df)

    assert "location_city" not in result.columns
    assert "location_department" not in result.columns
    assert "location_region" not in result.columns