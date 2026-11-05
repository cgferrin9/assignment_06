
import requests

from .settings import (
    BASE_URL,
    LLM_MODEL,
    LLM_REASONING_EFFORT,
    get_api_key,
    subject_questions,
)


def get_sentiment(text: str) -> dict:
    """Analyze the sentiment of text using the Azure API."""
    url = f"{BASE_URL}/api/azure/sentiment"
    headers = {"X-API-KEY": get_api_key()}
    response = requests.post(url, headers=headers, data={"text": text})
    response.raise_for_status()
    return response.json()


def get_entities(text: str) -> dict:
    url = f"{BASE_URL}/api/azure/entityrecognition"
    headers = {"X-API-KEY": get_api_key()}
    response = requests.post(url, headers=headers, data={"text": text})
    response.raise_for_status()
    return response.json()


def classify_mentions(review_text: str, mentions: list) -> dict:
    url = f"{BASE_URL}/api/ollama/systemone"
    headers = {"X-API-KEY": get_api_key()}

    body = {
        "state": {"review": review_text},
        "questions": subject_questions(mentions)
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()


def ask_llm(messages: list) -> str:
    url = f"{BASE_URL}/api/azure/foundry/chat/completions"
    headers = {"X-API-KEY": get_api_key()}

    params = {
        "model": LLM_MODEL,
        "reasoning_effort": LLM_REASONING_EFFORT,
    }

    response = requests.post(
        url,
        headers=headers,
        json=messages,
        params=params,
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
