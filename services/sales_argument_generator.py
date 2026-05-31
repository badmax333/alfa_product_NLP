"""Генерация sales-аргументов через Mistral."""

import json
import re
from typing import Any

from config.metrics import PORTRAIT_BEHAVIORAL_PROFILES
from config.sales_arguments import INTERACTION_TYPES
from services.llm import MISTRAL_MODEL, get_mistral_client
from services.sales_arg_renderer import render_sales_arg_prompt

_INTERACTION_TYPES_BY_ID = {t["id"]: t for t in INTERACTION_TYPES}


def _extract_json(text: str) -> dict[str, Any]:
    """Извлекает JSON из ответа LLM, даже если модель вернула markdown-блок."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])

    raise ValueError("JSON не найден в ответе модели")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def generate_sales_argument(
    classification: dict[str, Any],
    interaction_type: str,
    client_features: dict[str, Any],
) -> dict[str, Any]:
    """
    Рендерит промпт, вызывает Mistral и возвращает sales-аргумент
    в формате, совместимом с UI и генератором метрик.
    """
    itype_meta = _INTERACTION_TYPES_BY_ID.get(interaction_type)
    if not itype_meta:
        raise ValueError("Неизвестный тип взаимодействия")

    rendered_prompt = render_sales_arg_prompt(
        classification=classification,
        interaction_type=interaction_type,
        client_features=client_features,
    )

    client = get_mistral_client()
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": rendered_prompt}],
        temperature=0.7,
        max_tokens=1200,
    )
    message = response.choices[0].message
    raw_text = str(message.content) if message and message.content else ""
    parsed = _extract_json(raw_text)

    portrait_id = classification["predicted_class"]
    profile = PORTRAIT_BEHAVIORAL_PROFILES.get(portrait_id, {})
    product = classification.get("recommended_product", {})

    return {
        "id": f"llm_{interaction_type}_{portrait_id.lower()}",
        "interaction_type": interaction_type,
        "channel": itype_meta["channel"],
        "portrait": portrait_id,
        "portrait_label": profile.get("name", classification.get("class_description", "")),
        "product_ame": product.get("ame"),
        "product_name": product.get("name", ""),
        "headline": _clean_text(parsed.get("headline")),
        "body": _clean_text(parsed.get("body")),
        "cta": _clean_text(parsed.get("cta")),
        "note": (
            "Сгенерировано Mistral на основе портрета клиента, рекомендованного продукта, "
            "Top-5 SHAP-признаков и выбранного формата взаимодействия."
        ),
        "rendered_prompt": rendered_prompt,
        "raw_llm_response": raw_text,
    }
