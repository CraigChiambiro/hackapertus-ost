
import os

import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("CSCS_INFERENCE_API_KEY")
BASE_URL = os.getenv("CSCS_API_BASE_URL")
MODEL = os.getenv("APERTUS_MODEL")


def classify_claim(claim: str, reference: str) -> dict:
    missing = [
        name for name, value in (
            ("CSCS_INFERENCE_API_KEY", API_KEY),
            ("CSCS_API_BASE_URL", BASE_URL),
            ("APERTUS_MODEL", MODEL),
        )
        if not value or not value.strip()
    ]
    if missing:
        raise RuntimeError(
            "Missing required API configuration: "
            + ", ".join(missing)
            + ". Configure these variables in the environment or .env file."
        )

    if not BASE_URL.startswith(("https://", "http://")):
        raise RuntimeError(
            "CSCS_API_BASE_URL must start with https:// or http://."
        )
    prompt = f"""You are performing Natural Language Inference (NLI).

Determine the relationship between the CLAIM and the REFERENCE.

Return exactly one label:
0 = entailment
1 = neutral
2 = contradiction

Do not use outside knowledge.
Judge the claim only against the information contained in the reference.

CLAIM:
{claim}

REFERENCE:
{reference}

Return only the label number: 0, 1, or 2.
"""

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0,
            "max_tokens": 5,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()
    output = data["choices"][0]["message"]["content"].strip()

    return {
        "raw_output": output,
        "usage": data.get("usage"),
    }
