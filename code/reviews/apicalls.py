"""
apicalls.py — Step 1: one function per API endpoint.

Every function here does the same four things, the algorithm from lesson 4-1:

    1. build the URL from BASE_URL and the endpoint's path
    2. send the request with your key in the X-API-KEY header
    3. response.raise_for_status() — turn a 4xx/5xx into an exception, right here
    4. return response.json()

`raise_for_status()` is the line people leave out, and it is the one that saves
you an hour: without it, a bad key or a typo in the path comes back as a
perfectly normal-looking dictionary of error text, and your code fails three
steps later with a KeyError that points nowhere near the real problem.

These functions do not loop, do not build DataFrames, and do not cache. They make
one call and hand back what the API said. Everything else is Step 2's job.

This step is walked through line by line. The next two steps give you less.
"""

import requests

from .settings import (
    BASE_URL,
    LLM_MODEL,
    LLM_REASONING_EFFORT,
    get_api_key,
    subject_questions,
)


def get_sentiment(text: str) -> dict:
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
