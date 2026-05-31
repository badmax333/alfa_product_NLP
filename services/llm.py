"""Mistral API клиент."""

import os

from mistralai.client import Mistral

_client: Mistral | None = None


def get_mistral_client() -> Mistral:
    global _client
    if _client is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY не задан. Добавьте его в .env или переменные окружения."
            )
        _client = Mistral(api_key=api_key)
    return _client


MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
