"""Повторный цикл онбординга после двух неуспешных касаний."""

from __future__ import annotations

import json
import re
from typing import Any

from config.metrics import PORTRAIT_BEHAVIORAL_PROFILES
from config.propensity import PROPENSITY_PRODUCTS
from config.sales_arguments import INTERACTION_TYPES
from services.llm import call_mistral_messages
from services.propensity_feature_generator import generate_propensity_features
from services.propensity_scorer import score_propensity

SYSTEM_PROMPT = """
Ты генерируешь следующий sales-аргумент в закольцованном onboarding-пайплайне банка.

Контекст: клиент уже прошёл сегментацию, видел два предложения, по ним были собраны
метрики реакции, но активации не произошло. Нужно начать следующий цикл не с нуля,
а с учетом всей истории: портрет, признаки, скоринг склонности, показанные продукты,
показанные баннеры/аргументы и реакции клиента.

Правила:
- Верни только валидный JSON без markdown.
- Не повторяй продукты, заголовки и углы аргумента, которые уже показывали.
- Если прошлый интерес был низким, снизь давление и выбери более мягкую пользу.
- Если интерес был средним/высоким, убери главное сомнение и предложи конкретный следующий шаг.
- Говори про бизнес клиента, а не про внутренние банковские процессы.
- Не гарантируй прибыль, доходность или результат.
""".strip()

_INTERACTION_TYPES_BY_ID = {t["id"]: t for t in INTERACTION_TYPES}


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])

    raise ValueError("JSON не найден в ответе модели")


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _product_id_by_name(product_name: str | None) -> str | None:
    if not product_name:
        return None
    for product_id, meta in PROPENSITY_PRODUCTS.items():
        if meta.get("name") == product_name:
            return product_id
    return None


def _shown_product_ids(
    classification: dict[str, Any],
    stage1_argument: dict[str, Any] | None,
    stage2_argument: dict[str, Any] | None,
    selected_stage2_product: dict[str, Any] | None,
) -> list[str]:
    ids: list[str] = []

    stage1_product_id = _product_id_by_name(
        (stage1_argument or {}).get("product_name")
        or classification.get("recommended_product", {}).get("name")
    )
    if stage1_product_id:
        ids.append(stage1_product_id)

    stage2_product_id = (selected_stage2_product or {}).get(
        "product_id"
    ) or _product_id_by_name((stage2_argument or {}).get("product_name"))
    if stage2_product_id:
        ids.append(stage2_product_id)

    return list(dict.fromkeys(ids))


def build_onboarding_history(
    classification: dict[str, Any],
    stage1_argument: dict[str, Any] | None,
    stage1_metrics: dict[str, Any] | None,
    propensity_result: dict[str, Any] | None,
    stage2_argument: dict[str, Any] | None,
    stage2_metrics: dict[str, Any] | None,
    selected_stage2_product: dict[str, Any] | None,
) -> dict[str, Any]:
    """Собрать компактную историю двух касаний для LLM и нового скоринга."""
    shown_ids = _shown_product_ids(
        classification=classification,
        stage1_argument=stage1_argument,
        stage2_argument=stage2_argument,
        selected_stage2_product=selected_stage2_product,
    )
    return {
        "activation_status": "not_activated_after_two_attempts",
        "classification": classification,
        "attempts": [
            {
                "stage": "stage1",
                "argument": stage1_argument or {},
                "metrics": stage1_metrics or {},
                "activated": False,
            },
            {
                "stage": "stage2",
                "argument": stage2_argument or {},
                "metrics": stage2_metrics or {},
                "selected_product": selected_stage2_product or {},
                "activated": False,
            },
        ],
        "previous_propensity": propensity_result or {},
        "shown_product_ids": shown_ids,
        "shown_headlines": [
            item.get("headline")
            for item in (stage1_argument or {}, stage2_argument or {})
            if item.get("headline")
        ],
    }


def _select_next_product(
    scored_products: list[dict[str, Any]], shown_product_ids: list[str]
) -> dict[str, Any]:
    for product in scored_products:
        if product.get("product_id") not in shown_product_ids:
            return product
    if not scored_products:
        raise ValueError("Нет продуктов для нового цикла")
    return scored_products[0]


def render_recycle_prompt(
    classification: dict[str, Any],
    client_features: dict[str, Any],
    onboarding_history: dict[str, Any],
    next_product: dict[str, Any],
    interaction_type: str,
) -> str:
    profile = PORTRAIT_BEHAVIORAL_PROFILES.get(classification["predicted_class"], {})
    payload = {
        "task": "generate_next_cycle_sales_argument",
        "required_output": {
            "headline": "короткий заголовок по каналу",
            "body": "текст аргумента без воды",
            "cta": "короткий призыв к действию",
            "strategy_note": "1 предложение: почему выбран такой угол",
        },
        "client": {
            "portrait": classification,
            "behavioral_profile": profile,
            "known_features": client_features,
        },
        "history": onboarding_history,
        "next_offer": {
            "interaction_type": interaction_type,
            "interaction_type_label": _INTERACTION_TYPES_BY_ID.get(
                interaction_type, {}
            ).get("label", interaction_type),
            "product": next_product,
        },
        "instructions": [
            "Считать, что оба предыдущих предложения не привели к активации.",
            "Не использовать shown_product_ids и shown_headlines как новый оффер/заголовок.",
            "Опирайся на top_factors выбранного продукта и последние метрики реакции.",
            "Для banner: headline до 10 слов, body 1-2 предложения, cta 2-3 слова.",
            "Для push: headline до 7 слов, body до 150 символов.",
            "Для voice: короткий скрипт с признанием прошлых касаний и новым углом.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def generate_recycle_onboarding(
    classification: dict[str, Any],
    client_features: dict[str, Any],
    stage1_argument: dict[str, Any] | None,
    stage1_metrics: dict[str, Any] | None,
    propensity_result: dict[str, Any] | None,
    stage2_argument: dict[str, Any] | None,
    stage2_metrics: dict[str, Any] | None,
    selected_stage2_product: dict[str, Any] | None,
    interaction_type: str = "banner",
    top_k: int = 3,
) -> dict[str, Any]:
    """Запустить следующий onboarding-цикл с учетом всей накопленной истории."""
    itype_meta = _INTERACTION_TYPES_BY_ID.get(interaction_type)
    if not itype_meta:
        raise ValueError("Неизвестный тип взаимодействия")

    history = build_onboarding_history(
        classification=classification,
        stage1_argument=stage1_argument,
        stage1_metrics=stage1_metrics,
        propensity_result=propensity_result,
        stage2_argument=stage2_argument,
        stage2_metrics=stage2_metrics,
        selected_stage2_product=selected_stage2_product,
    )

    feature_generation = generate_propensity_features(
        classification=classification,
        client_features=client_features,
        metrics_result=stage2_metrics,
        sales_argument=stage2_argument,
        onboarding_history=history,
    )
    scoring = score_propensity(
        classification=classification,
        client_features=client_features,
        metrics_result=stage2_metrics,
        top_k=10,
        generated_features=feature_generation["features"],
    )

    shown_product_ids = history["shown_product_ids"]
    filtered_products = [
        product
        for product in scoring["all_products"]
        if product.get("product_id") not in shown_product_ids
    ]
    next_product = _select_next_product(scoring["all_products"], shown_product_ids)
    rendered_prompt = render_recycle_prompt(
        classification=classification,
        client_features=feature_generation["features"],
        onboarding_history=history,
        next_product=next_product,
        interaction_type=interaction_type,
    )
    raw_text = (
        call_mistral_messages(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": rendered_prompt},
            ],
            temperature=0.55,
            max_tokens=1600,
        )
        or ""
    )
    parsed = _extract_json(raw_text)

    argument = {
        "id": f"llm_recycle_{interaction_type}_{classification['predicted_class'].lower()}_{next_product.get('product_id')}",
        "interaction_type": interaction_type,
        "channel": itype_meta.get("channel", "digital"),
        "portrait": classification["predicted_class"],
        "portrait_label": profile_name(classification),
        "product_ame": next_product.get("product_ame"),
        "product_name": next_product.get("product_name", ""),
        "headline": _clean_text(parsed.get("headline")),
        "body": _clean_text(parsed.get("body")),
        "cta": _clean_text(parsed.get("cta")),
        "note": _clean_text(parsed.get("strategy_note"))
        or "Новый цикл: аргумент построен с учетом двух неуспешных касаний.",
        "propensity_score": next_product.get("propensity_score"),
        "rendered_prompt": rendered_prompt,
        "raw_llm_response": raw_text,
    }

    return {
        "cycle_number": 3,
        "activation_status": "not_activated_after_two_attempts",
        "shown_product_ids": shown_product_ids,
        "feature_generation": feature_generation,
        "propensity": {
            **scoring,
            "top_products": filtered_products[
                : max(1, min(int(top_k), len(filtered_products) or 1))
            ],
        },
        "selected_product": next_product,
        "next_argument": argument,
        "rendered_prompt": rendered_prompt,
        "system_prompt": SYSTEM_PROMPT,
        "raw_llm_response": raw_text,
    }


def profile_name(classification: dict[str, Any]) -> str:
    profile = PORTRAIT_BEHAVIORAL_PROFILES.get(classification["predicted_class"], {})
    return profile.get("name", classification.get("class_description", ""))
