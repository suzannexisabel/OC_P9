"""Nettoyage des événements OpenAgenda.

Ce module transforme les données brutes récupérées depuis OpenAgenda en un
DataFrame propre, dédupliqué et prêt pour la construction des documents RAG.

Utilisation depuis le projet :
    from src.data.clean_events import clean_events
    df_clean = clean_events(pd.DataFrame(events))

Exécution complète :
    python -m src.data.clean_events
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# Colonnes brutes réellement utilisées par le pipeline.
# Les autres colonnes reçues depuis l'API sont ignorées.
RAW_COLUMNS_TO_KEEP = [
    "uid",
    "title",
    "description",
    "longDescription",
    "keywords",
    "dateRange",
    "createdAt",
    "updatedAt",
    "timings",
    "firstTiming",
    "lastTiming",
    "nextTiming",
    "image",
    "attendanceMode",
    "status",
    "registration",
    "location",
    "age",
    "accessibility",
    "conditions",
    "billetterie",
    "links",
    "agenda_uid",
    "agenda_title",
]


ACCESSIBILITY_MAPPING = {
    "hi": "handicap auditif",
    "vi": "handicap visuel",
    "pi": "handicap psychique",
    "mi": "handicap moteur",
    "ii": "handicap intellectuel",
}

ATTENDANCE_MODE_MAPPING = {
    1: "Présentiel",
    2: "En ligne",
    3: "Hybride",
}

STATUS_MAPPING = {
    1: "Programmé",
    2: "Reprogrammé",
    3: "Déplacé en ligne",
    4: "Reporté",
    5: "Complet",
    6: "Annulé",
}

REGISTRATION_TYPE_MAPPING = {
    "email": "Email",
    "phone": "Téléphone",
    "link": "Lien",
}


def _is_missing(value: Any) -> bool:
    """Indique si une valeur simple est absente."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() == "missing value"
    if isinstance(value, (list, dict)):
        return len(value) == 0
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normalise_empty_values(value: Any) -> Any:
    """Convertit listes, dictionnaires et chaînes vides en None."""
    return None if _is_missing(value) else value


def _serialise_for_comparison(value: Any) -> str:
    """Rend une liste ou un dictionnaire comparable lors de la fusion."""
    if isinstance(value, (list, dict)):
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    return str(value).strip()


def _merge_duplicate_events(df: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les lignes ayant le même UID.

    La ligne la plus récemment mise à jour sert de base. Les listes sont
    réunies sans doublons, les dictionnaires sont complétés et les valeurs
    simples manquantes sont récupérées depuis les autres occurrences.
    """
    if "uid" not in df.columns:
        raise ValueError("La colonne obligatoire 'uid' est absente.")

    work = df.copy()

    if "updatedAt" in work.columns:
        work["updatedAt"] = pd.to_datetime(
            work["updatedAt"],
            errors="coerce",
            utc=True,
        )

    merged_rows: list[pd.Series] = []

    for _, group in work.groupby("uid", sort=False, dropna=False):
        if "updatedAt" in group.columns:
            group = group.sort_values("updatedAt", ascending=False)

        row = group.iloc[0].copy()

        for column in work.columns:
            if column in {"uid", "agenda_uid", "agenda_title"}:
                continue

            values = group[column].tolist()

            if any(isinstance(value, list) for value in values):
                unique_items: dict[str, Any] = {}

                for value in values:
                    if not isinstance(value, list):
                        continue
                    for item in value:
                        key = _serialise_for_comparison(item)
                        unique_items[key] = item

                row[column] = list(unique_items.values()) or None

            elif any(isinstance(value, dict) for value in values):
                merged_dict: dict[str, Any] = {}

                # Les données les plus récentes ont priorité.
                for value in reversed(values):
                    if isinstance(value, dict):
                        merged_dict.update(
                            {
                                key: content
                                for key, content in value.items()
                                if not _is_missing(content)
                            }
                        )

                row[column] = merged_dict or None

            elif _is_missing(row[column]):
                row[column] = next(
                    (
                        value
                        for value in values
                        if not _is_missing(value)
                    ),
                    None,
                )

        if "agenda_uid" in group.columns:
            row["agenda_uids"] = list(
                dict.fromkeys(group["agenda_uid"].dropna().tolist())
            ) or None
        else:
            row["agenda_uids"] = None

        if "agenda_title" in group.columns:
            row["agenda_titles"] = list(
                dict.fromkeys(
                    str(title).strip()
                    for title in group["agenda_title"].dropna().tolist()
                    if str(title).strip()
                )
            ) or None
        else:
            row["agenda_titles"] = None

        merged_rows.append(row)

    return pd.DataFrame(merged_rows).reset_index(drop=True)


def _clean_text(value: Any) -> str | None:
    """Nettoie un texte tout en conservant le Markdown et les emojis."""
    if not isinstance(value, str):
        return None if _is_missing(value) else str(value)

    text = re.sub(r"<[^>]+>", " ", value)
    text = re.sub(r"[-_]{3,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text or None


def _join_simple_list(value: Any) -> str | None:
    if not isinstance(value, list):
        return None if _is_missing(value) else str(value)

    items = [
        str(item).strip()
        for item in value
        if not _is_missing(item)
    ]
    return ", ".join(dict.fromkeys(items)) or None


def _format_datetime(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return None

    return parsed.strftime("%d/%m/%Y à %Hh%M")


def _format_timing(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None

    begin = _format_datetime(value.get("begin"))
    end = _format_datetime(value.get("end"))

    if begin and end:
        return f"Du {begin} au {end}"
    if begin:
        return f"À partir du {begin}"
    if end:
        return f"Jusqu'au {end}"

    return None


def _format_all_timings(value: Any) -> str | None:
    if not isinstance(value, list):
        return None

    timings = [
        formatted
        for item in value
        if (formatted := _format_timing(item))
    ]
    return " | ".join(timings) or None


def _format_accessibility(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None

    labels = [
        ACCESSIBILITY_MAPPING[code]
        for code, present in value.items()
        if present and code in ACCESSIBILITY_MAPPING
    ]

    if labels:
        return ", ".join(labels)

    return "Aucune information d'accessibilité"


def _format_image(value: Any) -> str | None:
    """Retourne en priorité l'URL de l'image en taille complète."""
    if isinstance(value, str):
        return value.strip() or None

    if not isinstance(value, dict):
        return None

    base = value.get("base", "")
    variants = value.get("variants")

    if isinstance(variants, list):
        full_variant = next(
            (
                variant.get("filename")
                for variant in variants
                if isinstance(variant, dict)
                and variant.get("type") == "full"
                and variant.get("filename")
            ),
            None,
        )
        if full_variant:
            return f"{base}{full_variant}"

    filename = value.get("filename")
    if filename:
        return f"{base}{filename}"

    return None


def _format_registration(value: Any) -> str | None:
    if not isinstance(value, list):
        return None

    parts: list[str] = []

    for item in value:
        if not isinstance(item, dict):
            continue

        content = item.get("value")
        if _is_missing(content):
            continue

        raw_type = str(item.get("type", "")).strip().lower()
        label = REGISTRATION_TYPE_MAPPING.get(
            raw_type,
            raw_type.capitalize() or "Contact",
        )
        parts.append(f"{label} : {content}")

    return " | ".join(parts) or None


def _extract_localised_text(value: Any, language: str = "fr") -> str | None:
    if isinstance(value, str):
        return value.strip() or None

    if isinstance(value, dict):
        selected = value.get(language)
        if isinstance(selected, str) and selected.strip():
            return selected.strip()

        for candidate in value.values():
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

    return None


def _build_location_text(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None

    description = _extract_localised_text(value.get("description"))
    access = _extract_localised_text(value.get("access"))

    parts = [
        f"Lieu : {value.get('name')}" if value.get("name") else None,
        f"Adresse : {value.get('address')}" if value.get("address") else None,
        (
            f"Description du lieu : {description}"
            if description
            else None
        ),
        f"Accès : {access}" if access else None,
        f"Téléphone : {value.get('phone')}" if value.get("phone") else None,
        f"Email : {value.get('email')}" if value.get("email") else None,
        (
            f"Site web : {value.get('website')}"
            if value.get("website")
            else None
        ),
        (
            f"Image du lieu : {value.get('image')}"
            if value.get("image")
            else None
        ),
    ]

    return " | ".join(part for part in parts if part) or None


def _format_age(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None

    minimum = value.get("min")
    maximum = value.get("max")

    if minimum is not None and maximum is not None:
        return f"De {minimum} à {maximum} ans"
    if minimum is not None:
        return f"À partir de {minimum} ans"
    if maximum is not None:
        return f"Jusqu'à {maximum} ans"

    return None


def _format_links(value: Any) -> str | None:
    if not isinstance(value, list):
        return None

    parts: list[str] = []

    for item in value:
        if not isinstance(item, dict):
            continue

        link = item.get("link")
        data = item.get("data")
        data = data if isinstance(data, dict) else {}

        candidates = [
            f"Lien : {link}" if link else None,
            f"Titre : {data.get('title')}" if data.get("title") else None,
            f"Auteur : {data.get('author')}" if data.get("author") else None,
            (
                f"Source : {data.get('provider_name')}"
                if data.get("provider_name")
                else None
            ),
            (
                f"Description : {data.get('description')}"
                if data.get("description")
                else None
            ),
        ]

        parts.extend(candidate for candidate in candidates if candidate)

    return " | ".join(parts) or None


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les événements OpenAgenda et retourne un DataFrame prêt pour le RAG."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un pandas.DataFrame.")

    missing_required = {"uid", "title"} - set(df.columns)
    if missing_required:
        raise ValueError(
            f"Colonnes obligatoires absentes : {sorted(missing_required)}"
        )

    available_columns = [
        column
        for column in RAW_COLUMNS_TO_KEEP
        if column in df.columns
    ]
    work = df.loc[:, available_columns].copy()

    # Normalisation des objets vides avant la fusion.
    for column in work.columns:
        work[column] = work[column].apply(_normalise_empty_values)

    work = _merge_duplicate_events(work)

    # Textes principaux.
    if "description" in work.columns:
        work["description"] = work["description"].apply(_clean_text)

    if "longDescription" in work.columns:
        work["longDescription"] = work["longDescription"].apply(_clean_text)

    if "conditions" in work.columns:
        work["conditions"] = work["conditions"].apply(_clean_text)

    # Listes simples.
    if "keywords" in work.columns:
        work["keywords"] = work["keywords"].apply(_join_simple_list)

    if "agenda_titles" in work.columns:
        work["agenda_titles"] = work["agenda_titles"].apply(_join_simple_list)

    # Accessibilité, mode de participation et statut.
    if "accessibility" in work.columns:
        work["accessibility_text"] = work["accessibility"].apply(
            _format_accessibility
        )
        work.drop(columns="accessibility", inplace=True)

    if "attendanceMode" in work.columns:
        work["attendanceMode"] = work["attendanceMode"].map(
            ATTENDANCE_MODE_MAPPING
        )
        work.rename(
            columns={"attendanceMode": "mode_participation"},
            inplace=True,
        )

    if "status" in work.columns:
        work["status"] = work["status"].map(STATUS_MAPPING)

    # Créneaux.
    if "timings" in work.columns:
        work["timings"] = work["timings"].apply(_format_all_timings)

    for column in ["firstTiming", "lastTiming", "nextTiming"]:
        if column in work.columns:
            work[column] = work[column].apply(_format_timing)

    # Image et inscription.
    if "image" in work.columns:
        work["image"] = work["image"].apply(_format_image)

    if "registration" in work.columns:
        work["registration"] = work["registration"].apply(
            _format_registration
        )

    # Lieu : colonnes de filtres + texte riche pour le RAG.
    if "location" in work.columns:
        work["location_name"] = work["location"].apply(
            lambda value: value.get("name")
            if isinstance(value, dict)
            else None
        )
        work["location_address"] = work["location"].apply(
            lambda value: value.get("address")
            if isinstance(value, dict)
            else None
        )
        work["location_city"] = work["location"].apply(
            lambda value: value.get("city")
            if isinstance(value, dict)
            else None
        )
        work["location_postal_code"] = work["location"].apply(
            lambda value: value.get("postalCode")
            if isinstance(value, dict)
            else None
        )
        work["location_department"] = work["location"].apply(
            lambda value: value.get("department")
            if isinstance(value, dict)
            else None
        )
        work["location_region"] = work["location"].apply(
            lambda value: value.get("region")
            if isinstance(value, dict)
            else None
        )
        work["location_latitude"] = work["location"].apply(
            lambda value: value.get("latitude")
            if isinstance(value, dict)
            else None
        )
        work["location_longitude"] = work["location"].apply(
            lambda value: value.get("longitude")
            if isinstance(value, dict)
            else None
        )
        work["location_text"] = work["location"].apply(
            _build_location_text
        )
        work.drop(columns="location", inplace=True)

    # Uniformisation géographique validée dans le notebook.
    if "location_region" in work.columns:
        work["location_region"] = (
            work["location_region"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace(
                {
                    "occitanie": "Occitanie",
                    "occitania": "Occitanie",
                }
            )
        )

    if "location_department" in work.columns:
        work["location_department"] = (
            work["location_department"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace(
                {
                    "haute-garonne": "Haute-Garonne",
                    "alto garona": "Haute-Garonne",
                }
            )
        )

    if "age" in work.columns:
        work["age"] = work["age"].apply(_format_age)

    if "links" in work.columns:
        work["links"] = work["links"].apply(_format_links)

    if "billetterie" in work.columns:
        work["billetterie"] = work["billetterie"].apply(
            _normalise_empty_values
        )

    # Suppression des anciennes colonnes d'agenda, remplacées par les listes fusionnées.
    work.drop(
        columns=["agenda_uid", "agenda_title"],
        inplace=True,
        errors="ignore",
    )

    # Ordre final explicite.
    final_columns = [
        "uid",
        "title",
        "description",
        "longDescription",
        "keywords",
        "dateRange",
        "createdAt",
        "updatedAt",
        "timings",
        "firstTiming",
        "lastTiming",
        "nextTiming",
        "image",
        "mode_participation",
        "status",
        "registration",
        "age",
        "accessibility_text",
        "conditions",
        "billetterie",
        "links",
        "agenda_uids",
        "agenda_titles",
        "location_name",
        "location_address",
        "location_city",
        "location_postal_code",
        "location_department",
        "location_region",
        "location_latitude",
        "location_longitude",
        "location_text",
    ]

    # Suppression des colonnes constantes (une seule valeur non vide)
    constant_columns = []

    for col in work.columns:
        non_null_values = work[col].dropna()

        if non_null_values.empty:
            constant_columns.append(col)
            continue

        comparable_values = non_null_values.map(_serialise_for_comparison)

        if comparable_values.nunique() <= 1:
            constant_columns.append(col)

    # On conserve toujours les colonnes importantes
    protected_columns = {
        "uid",
        "title",
        "description",
        "longDescription",
        "keywords",
    }

    constant_columns = [
        col
        for col in constant_columns
        if col not in protected_columns
    ]

    work.drop(columns=constant_columns, inplace=True)

    final_columns = [
        col
        for col in final_columns
        if col in work.columns
    ]

    return work.loc[:, final_columns].reset_index(drop=True)


def save_clean_events(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Sauvegarde le DataFrame nettoyé en Parquet."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def run_pipeline(
    output_path: str | Path = "data/processed/events_clean.parquet",
) -> pd.DataFrame:
    """Importe, nettoie et sauvegarde les événements."""
    from src.data.openagenda_import import get_toulouse_events

    events = get_toulouse_events()
    raw_df = pd.DataFrame(events)
    clean_df = clean_events(raw_df)
    save_clean_events(clean_df, output_path)

    print(f"Événements bruts : {len(raw_df)}")
    print(f"Événements nettoyés et uniques : {len(clean_df)}")
    print(f"Fichier sauvegardé : {Path(output_path)}")

    return clean_df


if __name__ == "__main__":
    run_pipeline()