"""Mistral API клиент с автоматическим повтором при ошибках 429/403 и логированием."""

import os
import random
import time
from pathlib import Path

from mistralai.client import Mistral

_client: Mistral | None = None
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

# --- Параметры повтора при rate-limit ---
_MAX_RETRIES = 6  # максимум попыток
_BACKOFF_INITIAL = 10.0  # начальная пауза, секунды
_BACKOFF_MAX = 64.0  # максимальная пауза, секунды

# --- Минимальная пауза между запросами к LLM (сек) ---
_INTER_REQUEST_DELAY = float(os.getenv("MISTRAL_REQUEST_DELAY", "2.0"))

# --- Файл для логирования всех запросов и ответов ---
_LOG_FILE = Path("logs/llm_calls.log")


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


def _log_llm_call(
    prompt: str,
    response: str | None,
    attempt: int,
    error: str | None = None,
) -> None:
    """Дописывает запрос и ответ в лог-файл."""
    _LOG_FILE.parent.mkdir(exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    status = "OK" if error is None else "ERROR"
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"[{ts}] attempt={attempt} | model={MISTRAL_MODEL} | {status}\n")
        f.write(f"── PROMPT ({len(prompt)} chars) ──\n{prompt}\n")
        f.write("── RESPONSE ──\n")
        f.write(f"{response if response is not None else '[' + str(error) + ']'}\n")
        f.write("\n")


def _is_rate_limit(exc: Exception) -> bool:
    """Возвращает True, если исключение вызвано rate-limit или временным отказом (429, 403)."""
    msg = str(exc).lower()
    return (
        "429" in msg
        or "403" in msg
        or "rate limit" in msg
        or "too many requests" in msg
        or "ratelimit" in msg
        or "requests per" in msg
        or "forbidden" in msg
    )


def call_mistral_messages(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1200,
) -> str | None:
    """
    Отправляет запрос в Mistral Chat API и возвращает текст ответа.

    При получении ошибки 429 (превышен лимит запросов) функция автоматически
    ждёт и повторяет запрос с экспоненциальным backoff + случайным jitter:
      попытка 1 → ждёт ~2с
      попытка 2 → ждёт ~4с
      попытка 3 → ждёт ~8с  ... до _BACKOFF_MAX

    Все остальные ошибки (5xx, сетевые, ошибки JSON) выбрасываются сразу
    без повтора — они не связаны с rate limit.

    Args:
        messages: список сообщений Chat API.
        temperature: температура сэмплинга (0.0–1.0).
        max_tokens: максимальное количество токенов в ответе.

    Returns:
        Текст ответа от Mistral.

    Raises:
        RuntimeError: если MISTRAL_API_KEY не задан.
        Exception: если все _MAX_RETRIES попыток исчерпаны или ошибка не rate-limit.
    """
    client = get_mistral_client()
    backoff = _BACKOFF_INITIAL
    prompt_for_log = "\n---\n".join(
        f"[{m['role']}]\n{m.get('content', '')}" for m in messages
    )

    for attempt in range(_MAX_RETRIES):
        if _INTER_REQUEST_DELAY > 0:
            time.sleep(_INTER_REQUEST_DELAY)
        try:
            response = client.chat.complete(
                model=MISTRAL_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            message = response.choices[0].message
            result = str(message.content) if message and message.content else ""
            _log_llm_call(prompt_for_log, result, attempt + 1)
            return result

        except Exception as exc:
            if _is_rate_limit(exc) and attempt < _MAX_RETRIES - 1:
                wait = backoff + random.uniform(0.0, 1.0)
                print(
                    f"\n[mistral] rate-limit — жду {wait:.1f}с "
                    f"(попытка {attempt + 1}/{_MAX_RETRIES})",
                    flush=True,
                )
                _log_llm_call(prompt_for_log, None, attempt + 1, error=str(exc))
                time.sleep(wait)
                backoff = min(backoff * 2, _BACKOFF_MAX)
            else:
                _log_llm_call(prompt_for_log, None, attempt + 1, error=str(exc))
                raise


def call_mistral(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1200,
) -> str | None:
    """Отправляет одиночный user prompt в Mistral с retry/backoff."""
    return call_mistral_messages(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
