import requests
import os
from dotenv import load_dotenv

import json

load_dotenv()

api_key = os.environ.get("OPENAGENDA_KEY")

headers = {
    "key": api_key
}

response = requests.get("https://api.openagenda.com/v2/agendas", headers=headers)

print("Status :", response.status_code)

if response.status_code == 200:
    data = response.json()
    print(type(data))
    print(data.keys())
    print(json.dumps(data, indent=2, ensure_ascii=False))
else:
    print("Erreur :", response.text)

print("Clé trouvée :", api_key is not None)