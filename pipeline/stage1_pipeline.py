"""
Пайплайн для генерации синтетических записей ступени 1.

Используется для батч-генерации данных, экспериментов и оценки качества
sales-аргументов и метрик взаимодействия.

Пример использования:
    # Одиночный запуск (случайный клиент, без LLM):
    from pipeline.stage1_pipeline import run_single
    result = run_single(method="random")

    # Батч-генерация (10 клиентов, без LLM):
    from pipeline.stage1_pipeline import run_batch
    results = run_batch(n=10, method="random")

    # Запуск через CLI (5 случайных клиентов):
    python -m pipeline.stage1_pipeline
"""

import random
from typing import Any

from config.sales_arguments import INTERACTION_TYPES, MOCK_ARGUMENTS_BY_TYPE, MOCK_SALES_ARGUMENTS
from config.stage1 import DEFAULT_FEATURES, DEMO_PRESETS, EDITABLE_FEATURES
from models.classifier import predict
from services.metrics_generator import generate_metrics
from services.random_metrics_generator import generate_metrics_random
from services.sales_argument_generator import generate_sales_argument


def generate_random_client_features() -> dict[str, Any]:
    """
    Случайный профиль клиента: один из 5 пресетов с шумом по числовым полям.
    Возвращает только EDITABLE_FEATURES (8 признаков для классификатора).
    """
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


def run_single(
    client_features: dict[str, Any] | None = None,
    interaction_type: str | None = None,
    channel: str | None = None,
    method: str = "random",
) -> dict[str, Any]:
    """
    Полный пайплайн ступени 1 для одного клиента.

    Args:
        client_features: 8 редактируемых признаков; если None — генерируются случайно.
        interaction_type: id типа взаимодействия ('banner'|'push'|'voice');
                          если None — выбирается случайно.
        channel: 'digital' | 'voice'; если None — берётся из interaction_type.
        method: 'random' — случайная генерация метрик (без LLM),
                'llm' — через Mistral API (требует MISTRAL_API_KEY в .env).

    Returns:
        dict с ключами: client_features, classification, interaction_type,
        sales_argument, channel, metrics_result.
    """
    if client_features is None:
        client_features = generate_random_client_features()

    classification = predict(client_features)

    if interaction_type is None:
        interaction_type = random.choice([t["id"] for t in INTERACTION_TYPES])

    itype_meta = next((t for t in INTERACTION_TYPES if t["id"] == interaction_type), None)
    if channel is None:
        channel = itype_meta["channel"] if itype_meta else "digital"

    if method == "llm":
        sales_argument = generate_sales_argument(
            classification=classification,
            interaction_type=interaction_type,
            client_features=client_features,
        )
        metrics_result = generate_metrics(
            classification=classification,
            sales_argument=sales_argument,
            channel=channel,
            client_features=client_features,
        )
    else:
        sales_argument = MOCK_ARGUMENTS_BY_TYPE.get(interaction_type, MOCK_SALES_ARGUMENTS[0])
        metrics_result = generate_metrics_random(
            classification=classification,
            sales_argument=sales_argument,
            channel=channel,
            client_features=client_features,
        )

    return {
        "client_features": client_features,
        "classification": classification,
        "interaction_type": interaction_type,
        "sales_argument": sales_argument,
        "channel": channel,
        "metrics_result": metrics_result,
    }


def run_batch(n: int = 10, method: str = "random") -> list[dict[str, Any]]:
    """Запустить пайплайн для n клиентов и вернуть список результатов."""
    return [run_single(method=method) for _ in range(n)]


if __name__ == "__main__":
    print("=== Stage 1 Pipeline — Batch Generation Demo (n=5, method=random) ===\n")
    results = run_batch(n=5, method="random")
    for i, r in enumerate(results, 1):
        cl = r["classification"]
        mr = r["metrics_result"]
        key_metric = next(
            (
                m
                for m in mr["metrics"]
                if m["name"] in ("product_activated", "product_connected_after_call")
            ),
            None,
        )
        activated = key_metric["value"] if key_metric else "?"
        print(
            f"[{i}] Portrait={cl['predicted_class']}"
            f"  channel={r['channel']:<8}"
            f"  interest={mr['interest_score']:.2f}"
            f"  product=AME-{cl['recommended_product']['ame']}"
            f"  activated={activated}"
        )
    print(f"\nTotal: {len(results)} records generated")
