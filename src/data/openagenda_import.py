import requests
import os
from dotenv import load_dotenv

from datetime import datetime
from dateutil.relativedelta import relativedelta

import json

# ----------------------

load_dotenv()

API_KEY = os.environ.get("OPENAGENDA_KEY")
BASE_URL = "https://api.openagenda.com/v2"

HEADERS = {
    "key": API_KEY
}

# Fonction pour identifier les agendas à Toulouse 

def get_toulouse_agendas():
    url = f"{BASE_URL}/agendas"

    all_agendas = []
    after = None

    while True:
        params = {
            "search": "Toulouse",
            "size": 100,
            "sort": "createdAt.desc"
        }

        # À partir de la 2e requête, on ajoute le curseur "after"
        if after is not None:
            params["after"] = after

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        agendas = data.get("agendas", [])
        all_agendas.extend(agendas)

        print(
            f"{len(agendas)} agendas récupérés sur cette page "
            f"- total actuel : {len(all_agendas)} "
            f"- status={response.status_code} "    
        )

        # Curseur permettant de récupérer la page suivante
        after = data.get("after")

        # S'il n'y a plus de curseur, on a terminé
        if not after:
            break

    print(f"\nTotal : {len(all_agendas)} agendas liés à Toulouse")

    return all_agendas

# Fonction pour identifier les evenements à toulouse de - de 1an d'un agenda

def get_events_from_agenda(agenda_uid):
    url = f"{BASE_URL}/agendas/{agenda_uid}/events"

    # Date exacte d'il y a un an
    date_min = (
        datetime.now() - relativedelta(years=1)
    ).strftime("%Y-%m-%d")

    all_events = []
    after = None

    while True:
        params = {
            "adminLevel4": "Toulouse",
            "timings[gte]": date_min,
            "size": 300,
            "detailed": 1,
            "monolingual": "fr"
        }

        if after is not None:
            params["after"] = after

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        events = data.get("events", [])

        all_events.extend(events)

        print(
            f"{len(events)} événement(s) récupéré(s)"
            f"- total actuel : {len(all_events)}"
        )

        after = data.get("after")

        if not after:
            break

    print(
        f"Agenda {agenda_uid} : "
        f"{len(all_events)} événement(s) au total\n"
    )

    return all_events

# Fonction pour recuperer les evenement Toulousains de moins de 1 ans sur open agenda

def get_toulouse_events(
        max_agendas: int | None = None
):
    # 1. Récupération de tous les agendas liés à Toulouse
    agendas = get_toulouse_agendas()

    if max_agendas is not None:
        agendas = agendas[:max_agendas]

    all_events = []

    # 2. Récupération des événements de chaque agenda
    for i, agenda in enumerate(agendas, start=1):
        agenda_uid = agenda["uid"]
        agenda_title = agenda.get("title", "Sans titre")

        print(
            f"\n[{i}/{len(agendas)}] "
            f"Agenda : {agenda_title}"
        )

        try:
            events = get_events_from_agenda(agenda_uid)

            print(
                f"→ {len(events)} événement(s) récupéré(s) "
                f"dans cet agenda"
            )

            for event in events:
                event["agenda_uid"] = agenda_uid
                event["agenda_title"] = agenda_title

            all_events.extend(events)

            print(
                f"→ Total cumulé : "
                f"{len(all_events)} événement(s)"
            )

        except requests.RequestException as error:
            print(
                f"→ Erreur pour l'agenda {agenda_uid} : "
                f"{error}"
            )

    print(
        f"\nRécupération terminée : "
        f"{len(all_events)} événement(s) au total"
    )

    return all_events
