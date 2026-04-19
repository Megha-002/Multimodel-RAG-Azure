import requests
from config import HF_API_KEY, HF_EMBED_URL

def get_embedding(text: str):

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text}

    response = requests.post(
        HF_EMBED_URL,
        headers=headers,
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    result = response.json()

    # HF sometimes returns nested list — flatten it
    if isinstance(result[0], list):
        return result[0]
    return result