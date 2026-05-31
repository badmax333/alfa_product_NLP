"""Генерация sales-аргументов через Mistral (Stage 1 и Stage 2)."""

import json
import re
from typing import Any

from config.metrics import PORTRAIT_BEHAVIORAL_PROFILES
from config.sales_arguments import INTERACTION_TYPES
from services.llm import call_mistral
from services.sales_arg_renderer import (
    render_sales_arg_prompt,
    render_stage2_sales_arg_prompt,
)

_INTERACTION_TYPES_BY_ID = {t["id"]: t for t in INTERACTION_TYPES}


# ---------------------------------------------------------------------------
# Shared private helpers
# ---------------------------------------------------------------------------


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


def _build_argument_fields(
    parsed: dict[str, Any],
    interaction_type: str,
    portrait_id: str,
    product: dict[str, Any],
    itype_meta: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    return {
        "interaction_type": interaction_type,
        "channel": itype_meta.get("channel", "digital"),
        "portrait": portrait_id,
        "portrait_label": profile.get("name", ""),
        "product_ame": product.get("ame"),
        "product_name": product.get("name", ""),
        "headline": _clean_text(parsed.get("headline")),
        "body": _clean_text(parsed.get("body")),
        "cta": _clean_text(parsed.get("cta")),
    }


# ---------------------------------------------------------------------------
# Public: Stage 1
# ---------------------------------------------------------------------------


def generate_sales_argument(
    classification: dict[str, Any],
    interaction_type: str,
    client_features: dict[str, Any],
) -> dict[str, Any]:
    """Stage 1: рендерит промпт, вызывает Mistral, возвращает sales-аргумент."""
    itype_meta = _INTERACTION_TYPES_BY_ID.get(interaction_type)
    if not itype_meta:
        raise ValueError("Неизвестный тип взаимодействия")

    rendered_prompt = render_sales_arg_prompt(
        classification=classification,
        interaction_type=interaction_type,
        client_features=client_features,
    )
    raw_text = call_mistral(rendered_prompt)
    parsed = _extract_json(raw_text)

    portrait_id = classification["predicted_class"]
    profile = PORTRAIT_BEHAVIORAL_PROFILES.get(portrait_id, {})
    product = classification.get("recommended_product", {})

    result = _build_argument_fields(
        parsed, interaction_type, portrait_id, product, itype_meta, profile
    )
    result["id"] = f"llm_{interaction_type}_{portrait_id.lower()}"
    result["note"] = (
        "Сгенерировано Mistral на основе портрета клиента, рекомендованного продукта, "
        "Top-5 SHAP-признаков и выбранного формата взаимодействия."
    )
    result["rendered_prompt"] = rendered_prompt
    result["raw_llm_response"] = raw_text
    return result


# ---------------------------------------------------------------------------
# Public: Stage 2
# ---------------------------------------------------------------------------


def generate_stage2_argument(
    classification: dict[str, Any],
    interaction_type: str,
    client_features: dict[str, Any],
    propensity_product: dict[str, Any],
    stage1_argument: dict[str, Any] | None = None,
    stage1_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Stage 2: генерирует sales-аргумент с учётом истории взаимодействия и скоринга склонности.

    Args:
        classification: результат /api/v1/predict.
        interaction_type: 'banner' | 'push' | 'voice'.
        client_features: 8 редактируемых признаков клиента.
        propensity_product: один продукт из top_products результата score_propensity().
            Обязательные поля: product_id, product_name, product_ame, description, top_factors.
        stage1_argument: аргумент Ступени 1 (headline, body, cta, product_name, interaction_type).
            Если None — промпт строится без истории первого касания.
        stage1_metrics: результат генерации метрик (interest_score, user_reaction_text).
            Если None — тональность не корректируется по реакции.

    Returns:
        dict с полями id, interaction_type, channel, portrait, portrait_label,
        product_ame, product_name, headline, body, cta, note, propensity_score,
        rendered_prompt, raw_llm_response.
    """
    itype_meta = _INTERACTION_TYPES_BY_ID.get(interaction_type)
    if not itype_meta:
        raise ValueError("Неизвестный тип взаимодействия")

    rendered_prompt = render_stage2_sales_arg_prompt(
        classification=classification,
        interaction_type=interaction_type,
        client_features=client_features,
        propensity_product=propensity_product,
        stage1_argument=stage1_argument,
        stage1_metrics=stage1_metrics,
    )
    raw_text = call_mistral(rendered_prompt, max_tokens=1400)
    parsed = _extract_json(raw_text)

    portrait_id = classification["predicted_class"]
    profile = PORTRAIT_BEHAVIORAL_PROFILES.get(portrait_id, {})
    product = {
        "ame": propensity_product.get("product_ame"),
        "name": propensity_product.get("product_name", ""),
    }

    result = _build_argument_fields(
        parsed, interaction_type, portrait_id, product, itype_meta, profile
    )
    result["id"] = (
        f"llm_s2_{interaction_type}_{portrait_id.lower()}_{propensity_product.get('product_id', '')}"
    )
    result["propensity_score"] = propensity_product.get("propensity_score")

    interest = (stage1_metrics or {}).get("interest_score")
    interest_str = (
        f", интерес к Ступени 1: {interest:.2f}" if interest is not None else ""
    )
    result["note"] = (
        f"Stage 2. Продукт: {propensity_product.get('product_name', '')} "
        f"(склонность: {propensity_product.get('propensity_score', '—')}{interest_str}). "
        f"Факторы: {', '.join(f['feature'] for f in propensity_product.get('top_factors', [])[:3])}."
    )
    result["rendered_prompt"] = rendered_prompt
    result["raw_llm_response"] = raw_text
    return result
