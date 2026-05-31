"""LLM-генерация признаков клиента перед скорингом склонности."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config.stage1 import DEFAULT_FEATURES, FEATURE_LABELS
from services.llm import MISTRAL_MODEL, get_mistral_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_CONFIG_PATH = PROJECT_ROOT / "models" / "feature_config.json"

SYSTEM_PROMPT = """
Ты генерируешь признаки клиента для ML-модели склонности банковских продуктов.

Твоя задача: на основе демо-контекста клиента восстановить полный feature vector,
который правдоподобно описывает поведение клиента к моменту Stage 2.

Правила:
- Верни только валидный JSON без markdown.
- Не добавляй product_id: модель будет скорить один и тот же профиль по каждому продукту отдельно.
- priority_segment должен совпадать с predicted_class из классификатора.
- Сохраняй явно известные пользователем признаки, если они переданы во входном профиле.
- Остальные признаки сгенерируй согласованно с портретом, отраслью, sales-аргументом,
  реакцией клиента и метриками взаимодействия.
- Числовые признаки должны быть числами, бинарные признаки только 0 или 1.
- Категориальные признаки должны быть строками.
- Не выдумывай экстремальные значения без причины: профиль должен быть реалистичным для POC.
""".strip()


def _load_feature_config() -> dict[str, Any]:
    return json.loads(FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])

    raise ValueError("JSON не найден в ответе модели")


def _metric_values(metrics_result: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics_result:
        return {}
    return {
        item["name"]: item.get("value")
        for item in metrics_result.get("metrics", [])
        if isinstance(item, dict) and item.get("name")
    }


def _feature_schema(feature_config: dict[str, Any]) -> dict[str, Any]:
    cat_features = [name for name in feature_config["cat_features"] if name != "product_id"]
    num_features = feature_config["num_features"]
    binary_features = feature_config.get("binary_features", [])
    return {
        "cat_features": cat_features,
        "num_features": num_features,
        "binary_features": binary_features,
        "feature_labels": {
            name: FEATURE_LABELS.get(name, name)
            for name in cat_features + num_features
        },
    }


def render_propensity_feature_prompt(
    classification: dict[str, Any],
    client_features: dict[str, Any],
    metrics_result: dict[str, Any] | None,
    sales_argument: dict[str, Any] | None,
) -> str:
    """Собрать user prompt для генерации полного набора признаков Stage 2."""
    feature_config = _load_feature_config()
    schema = _feature_schema(feature_config)
    known_features = {**client_features, "priority_segment": classification["predicted_class"]}

    payload = {
        "task": "generate_propensity_features",
        "required_output": {
            "features": {
                "description": "dict со всеми cat_features без product_id и всеми num_features",
                "required_keys": schema["cat_features"] + schema["num_features"],
            },
            "reasoning_summary": "1-2 коротких предложения, почему такие признаки согласованы с контекстом",
        },
        "feature_schema": schema,
        "known_client_features": known_features,
        "classification": classification,
        "sales_argument": sales_argument or {},
        "interaction_metrics": {
            "interest_score": (metrics_result or {}).get("interest_score"),
            "user_reaction_text": (metrics_result or {}).get("user_reaction_text", ""),
            "metric_values": _metric_values(metrics_result),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _coerce_generated_features(
    parsed: dict[str, Any],
    feature_config: dict[str, Any],
    classification: dict[str, Any],
    client_features: dict[str, Any],
) -> dict[str, Any]:
    raw_features = parsed.get("features")
    if not isinstance(raw_features, dict):
        raise ValueError("Ответ модели должен содержать объект features")

    cat_features = [name for name in feature_config["cat_features"] if name != "product_id"]
    num_features = feature_config["num_features"]
    binary_features = set(feature_config.get("binary_features", []))
    required_features = cat_features + num_features

    features: dict[str, Any] = {}
    for name in required_features:
        if name in client_features:
            value = client_features[name]
        elif name == "priority_segment":
            value = classification["predicted_class"]
        elif name in raw_features:
            value = raw_features[name]
        elif name in DEFAULT_FEATURES:
            value = DEFAULT_FEATURES[name]
        else:
            raise ValueError(f"LLM не вернул обязательный признак {name}")

        if name in binary_features:
            features[name] = 1 if str(value).lower() in ("1", "true", "yes", "да") else 0
        elif name in num_features:
            try:
                features[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Признак {name} должен быть числом") from exc
        else:
            features[name] = str(value)

    return features


def generate_propensity_features(
    classification: dict[str, Any],
    client_features: dict[str, Any],
    metrics_result: dict[str, Any] | None,
    sales_argument: dict[str, Any] | None,
) -> dict[str, Any]:
    """Вызвать Mistral и вернуть полный набор признаков для модели склонности."""
    feature_config = _load_feature_config()
    rendered_prompt = render_propensity_feature_prompt(
        classification=classification,
        client_features=client_features,
        metrics_result=metrics_result,
        sales_argument=sales_argument,
    )

    client = get_mistral_client()
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rendered_prompt},
        ],
        temperature=0.35,
        max_tokens=2500,
    )
    message = response.choices[0].message
    raw_text = str(message.content) if message and message.content else ""
    parsed = _extract_json(raw_text)
    features = _coerce_generated_features(
        parsed=parsed,
        feature_config=feature_config,
        classification=classification,
        client_features=client_features,
    )

    return {
        "features": features,
        "reasoning_summary": str(parsed.get("reasoning_summary", "")).strip(),
        "rendered_prompt": rendered_prompt,
        "system_prompt": SYSTEM_PROMPT,
        "raw_llm_response": raw_text,
    }
