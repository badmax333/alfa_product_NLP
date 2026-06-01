"""
Полный двухступенчатый пайплайн для одного синтетического клиента.

Ступень 1: классификация → аргумент → метрики
Ступень 2: скоринг склонности → аргумент → метрики

Две стратегии генерации аргументов (задаются независимо для каждой ступени):
  personalized=True  — LLM создаёт аргумент под конкретный портрет и признаки клиента
  personalized=False — фиксированный обезличенный шаблон (без LLM, без персонализации)

Метод генерации метрик (не зависит от стратегии аргумента):
  metrics_method="llm"    — Mistral симулирует реакцию клиента; учитывает качество аргумента
  metrics_method="random" — правило-базированная симуляция по профилю портрета; аргумент игнорируется

Пример использования:
    from pipeline.full_pipeline import full_run_single, generate_random_client_features

    result = full_run_single(metrics_method="random")            # быстро, без LLM
    result = full_run_single(s1_personalized=True, metrics_method="llm")  # полный LLM-режим
"""

import random
from typing import Any

from config.sales_arguments import INTERACTION_TYPES
from config.stage1 import DEFAULT_FEATURES, DEMO_PRESETS, EDITABLE_FEATURES
from models.classifier import predict
from services.metrics_generator import generate_metrics
from services.propensity_scorer import score_propensity
from services.random_metrics_generator import generate_metrics_random
from services.sales_argument_generator import (
    generate_sales_argument,
    generate_stage2_argument,
)

# ---------------------------------------------------------------------------
# Обезличенные (generic) шаблоны аргументов — без персонализации по портрету
# ---------------------------------------------------------------------------

_GENERIC_S1: dict[str, dict[str, Any]] = {
    "banner": {
        "id": "generic_banner",
        "interaction_type": "banner",
        "channel": "digital",
        "portrait": "generic",
        "portrait_label": "Без персонализации",
        "product_ame": None,
        "product_name": "Банковские продукты",
        "headline": "Новые возможности для вашего бизнеса",
        "body": (
            "Альфа-Банк предлагает широкую линейку продуктов для малого и среднего бизнеса: "
            "расчётный счёт, эквайринг, кредитование и многое другое."
        ),
        "cta": "Узнать подробнее",
        "note": "Генерик-баннер без персонализации по портрету клиента",
        "rendered_prompt": "[generic — no LLM]",
        "raw_llm_response": "",
    },
    "push": {
        "id": "generic_push",
        "interaction_type": "push",
        "channel": "digital",
        "portrait": "generic",
        "portrait_label": "Без персонализации",
        "product_ame": None,
        "product_name": "Банковские продукты",
        "headline": "Специальное предложение от Альфа-Банка",
        "body": "Новые продукты для вашего бизнеса — ознакомьтесь с условиями.",
        "cta": "Подробнее",
        "note": "Генерик-push без персонализации",
        "rendered_prompt": "[generic — no LLM]",
        "raw_llm_response": "",
    },
    "voice": {
        "id": "generic_voice",
        "interaction_type": "voice",
        "channel": "voice",
        "portrait": "generic",
        "portrait_label": "Без персонализации",
        "product_ame": None,
        "product_name": "Банковские продукты",
        "headline": "Голосовой скрипт: общее предложение",
        "body": (
            "[Открытие] Добрый день, [Имя]. Это Альфа-Банк, звоним с предложением.\n\n"
            "[Аргумент] У нас широкая линейка продуктов для бизнеса: расчётный счёт, "
            "эквайринг, кредитование, зарплатный проект — всё в одном банке.\n\n"
            "[Закрытие] Хотите узнать подробнее о каком-либо продукте?"
        ),
        "cta": "Оформить заявку",
        "note": "Генерик-голосовой скрипт без персонализации",
        "rendered_prompt": "[generic — no LLM]",
        "raw_llm_response": "",
    },
}


def _generic_s2_argument(
    s2_interaction_type: str,
    propensity_top: dict[str, Any],
) -> dict[str, Any]:
    product_name = propensity_top.get("product_name", "продукт")
    channel = _channel_from_itype(s2_interaction_type)
    return {
        "id": f"generic_s2_{s2_interaction_type}",
        "interaction_type": s2_interaction_type,
        "channel": channel,
        "portrait": "generic",
        "portrait_label": "Без персонализации",
        "product_ame": propensity_top.get("product_ame"),
        "product_name": product_name,
        "headline": f"Предложение: {product_name}",
        "body": (
            f"Подключите {product_name} — популярный продукт Альфа-Банка для бизнеса. "
            "Простое оформление, выгодные условия."
        ),
        "cta": "Подключить",
        "note": "Генерик Stage 2 без персонализации",
        "rendered_prompt": "[generic S2 — no LLM]",
        "raw_llm_response": "",
        "propensity_score": propensity_top.get("propensity_score"),
    }


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _channel_from_itype(itype: str) -> str:
    for t in INTERACTION_TYPES:
        if t["id"] == itype:
            return t["channel"]
    return "digital"


def _get_conversion(metrics_result: dict[str, Any], channel: str) -> bool:
    """Извлекает бинарный факт подключения продукта из результата метрик."""
    key = (
        "product_activated" if channel == "digital" else "product_connected_after_call"
    )
    for m in metrics_result.get("metrics", []):
        if m["name"] == key:
            return bool(m["value"])
    return False


def generate_random_client_features() -> dict[str, Any]:
    """Случайный профиль клиента: один из 5 демо-пресетов с шумом по числовым полям."""
    preset = random.choice(DEMO_PRESETS)
    features = {**DEFAULT_FEATURES, **preset["overrides"]}
    if "days_from_ogrn" in features:
        features["days_from_ogrn"] = max(
            1, int(features["days_from_ogrn"] * random.uniform(0.7, 1.4))
        )
    if "week_sum_transactions" in features:
        features["week_sum_transactions"] = max(
            1000, int(features["week_sum_transactions"] * random.uniform(0.5, 2.0))
        )
    return {k: features[k] for k in EDITABLE_FEATURES if k in features}


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------


def full_run_single(
    client_features: dict[str, Any] | None = None,
    s1_interaction_type: str | None = None,
    s2_interaction_type: str | None = None,
    s1_personalized: bool = True,
    s2_personalized: bool = True,
    metrics_method: str = "random",
) -> dict[str, Any]:
    """
    Полный двухступенчатый пайплайн для одного клиента.

    Args:
        client_features: 8 редактируемых признаков. Если None — генерируются случайно.
        s1_interaction_type: 'banner' | 'push' | 'voice' для Ступени 1. Если None — случайно.
        s2_interaction_type: 'banner' | 'push' | 'voice' для Ступени 2. Если None — случайно.
        s1_personalized: True → LLM создаёт аргумент С1; False → обезличенный шаблон.
        s2_personalized: True → LLM создаёт аргумент С2; False → обезличенный шаблон.
        metrics_method: 'llm' → Mistral симулирует реакцию; 'random' → правило-базированная симуляция.

    Returns:
        dict со всеми промежуточными результатами и плоским словарём 'summary'.
    """
    itype_ids = [t["id"] for t in INTERACTION_TYPES]
    if client_features is None:
        client_features = generate_random_client_features()
    if s1_interaction_type is None:
        s1_interaction_type = random.choice(itype_ids)
    if s2_interaction_type is None:
        s2_interaction_type = random.choice(itype_ids)

    s1_channel = _channel_from_itype(s1_interaction_type)
    s2_channel = _channel_from_itype(s2_interaction_type)

    # Stage 1: classify
    classification = predict(client_features)

    # Stage 1: argument
    llm_error_s1 = None
    if s1_personalized:
        try:
            s1_argument = generate_sales_argument(
                classification=classification,
                interaction_type=s1_interaction_type,
                client_features=client_features,
            )
        except Exception as exc:
            llm_error_s1 = str(exc)
            s1_argument = _GENERIC_S1[s1_interaction_type]
    else:
        s1_argument = _GENERIC_S1[s1_interaction_type]

    # Stage 1: metrics
    metrics_fn = (
        generate_metrics if metrics_method == "llm" else generate_metrics_random
    )
    s1_metrics = metrics_fn(
        classification=classification,
        sales_argument=s1_argument,
        channel=s1_channel,
        client_features=client_features,
    )

    # Stage 2: propensity scoring
    propensity = score_propensity(
        classification=classification,
        client_features=client_features,
        metrics_result=s1_metrics,
        top_k=3,
    )
    propensity_top = propensity["top_products"][0]

    # Stage 2: argument
    llm_error_s2 = None
    if s2_personalized:
        try:
            s2_argument = generate_stage2_argument(
                classification=classification,
                interaction_type=s2_interaction_type,
                client_features=client_features,
                propensity_product=propensity_top,
                stage1_argument=s1_argument,
                stage1_metrics=s1_metrics,
            )
        except Exception as exc:
            llm_error_s2 = str(exc)
            s2_argument = _generic_s2_argument(s2_interaction_type, propensity_top)
    else:
        s2_argument = _generic_s2_argument(s2_interaction_type, propensity_top)

    # Stage 2: metrics
    s2_metrics = metrics_fn(
        classification=classification,
        sales_argument=s2_argument,
        channel=s2_channel,
        client_features=client_features,
    )

    return {
        "client_features": client_features,
        "classification": classification,
        "s1_interaction_type": s1_interaction_type,
        "s1_channel": s1_channel,
        "s1_argument": s1_argument,
        "s1_metrics": s1_metrics,
        "propensity": propensity,
        "s2_interaction_type": s2_interaction_type,
        "s2_channel": s2_channel,
        "s2_argument": s2_argument,
        "s2_metrics": s2_metrics,
        "llm_errors": {"s1": llm_error_s1, "s2": llm_error_s2},
        "summary": {
            "portrait": classification["predicted_class"],
            "s1_itype": s1_interaction_type,
            "s2_itype": s2_interaction_type,
            "s1_interest": s1_metrics["interest_score"],
            "s2_interest": s2_metrics["interest_score"],
            "s1_activated": _get_conversion(s1_metrics, s1_channel),
            "s2_activated": _get_conversion(s2_metrics, s2_channel),
            "s1_product": s1_argument.get("product_name", "?"),
            "s2_product": s2_argument.get("product_name", "?"),
            "s1_personalized": s1_personalized,
            "s2_personalized": s2_personalized,
            "metrics_method": metrics_method,
            "llm_fallback": bool(llm_error_s1 or llm_error_s2),
        },
    }
