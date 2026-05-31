"""Генерация синтетических метрик взаимодействия через Mistral."""

import json
import random
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from config.metrics import (
    LEVEL_NAMES,
    PORTRAIT_BEHAVIORAL_PROFILES,
    get_metrics_for_channel,
)
from config.sales_arguments import INTERACTION_TYPES
from services.llm import MISTRAL_MODEL, get_mistral_client

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_jinja_env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), autoescape=False)

CHANNEL_LABELS = {"digital": "Цифровой канал", "voice": "Голосовой канал"}
INTERACTION_TYPE_MAP = {t["id"]: t["label"] for t in INTERACTION_TYPES}


def render_metrics_prompt(
    classification: dict[str, Any],
    sales_argument: dict[str, Any],
    channel: str,
    client_features: dict[str, Any],
) -> str:
    portrait_id = classification["predicted_class"]
    profile = PORTRAIT_BEHAVIORAL_PROFILES.get(portrait_id, {})
    metrics = get_metrics_for_channel(channel)

    metrics_by_level: dict[int, list[dict]] = {}
    for m in metrics:
        metrics_by_level.setdefault(m["level"], []).append(m)

    template = _jinja_env.get_template("stage1_metrics_generation.j2")
    return template.render(
        portrait_id=portrait_id,
        portrait_name=profile.get("name", classification.get("class_description", "")),
        digital_affinity=profile.get("digital_affinity", "medium"),
        voice_affinity=profile.get("voice_affinity", "medium"),
        interest_base=profile.get("interest_base", 0.60),
        typical_behavior=profile.get("typical_behavior", ""),
        negative_triggers=profile.get("negative_triggers", ""),
        top_features=classification.get("top5_feature_importance", []),
        client_features=client_features,
        product=classification.get("recommended_product", {}),
        channel=channel,
        channel_label=CHANNEL_LABELS.get(channel, channel),
        interaction_type=sales_argument.get("interaction_type", ""),
        interaction_type_label=INTERACTION_TYPE_MAP.get(
            sales_argument.get("interaction_type", ""), sales_argument.get("interaction_type", "")
        ),
        sales_argument=sales_argument,
        metrics_by_level=metrics_by_level,
        level_names=LEVEL_NAMES,
    )


def _apply_noise(metrics: dict[str, Any], noise_factor: float = 0.15) -> dict[str, Any]:
    """Добавляет небольшой случайный шум к числовым метрикам (±noise_factor)."""
    noisy = {}
    for k, v in metrics.items():
        if isinstance(v, float):
            delta = v * noise_factor * (random.random() * 2 - 1)
            noisy[k] = round(max(0.0, v + delta), 1)
        elif isinstance(v, int) and v not in (0, 1):
            delta = max(1, int(v * noise_factor))
            noisy[k] = max(0, v + random.randint(-delta, delta))
        else:
            noisy[k] = v
    return noisy


def _extract_json(text: str) -> dict[str, Any]:
    """Извлекает JSON из ответа LLM (модель может обернуть его в маркдаун)."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])
    raise ValueError("JSON не найден в ответе модели")


def generate_metrics(
    classification: dict[str, Any],
    sales_argument: dict[str, Any],
    channel: str,
    client_features: dict[str, Any],
) -> dict[str, Any]:
    """
    Рендерит промпт, вызывает Mistral и возвращает структуру с метриками.

    Returns:
        {
          "channel": str,
          "portrait": str,
          "rendered_prompt": str,
          "interest_score": float,
          "user_reaction_text": str,
          "metrics": list[dict],  # метрики с value
          "raw_llm_response": str,
        }
    """
    rendered_prompt = render_metrics_prompt(
        classification=classification,
        sales_argument=sales_argument,
        channel=channel,
        client_features=client_features,
    )

    client = get_mistral_client()
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": rendered_prompt}],
        temperature=0.7,
        max_tokens=2048,
    )
    message = response.choices[0].message
    raw_text = str(message.content) if message and message.content else ""

    parsed = _extract_json(raw_text)
    raw_metrics: dict[str, Any] = parsed.get("metrics", {})
    raw_metrics = _apply_noise(raw_metrics)

    metrics_definitions = get_metrics_for_channel(channel)
    metrics_out = []
    for m in metrics_definitions:
        value = raw_metrics.get(m["name"])
        metrics_out.append(
            {
                "name": m["name"],
                "label": m["label"],
                "level": m["level"],
                "level_name": m["level_name"],
                "type": m["type"],
                "unit": m["unit"],
                "description": m["description"],
                "value": value,
            }
        )

    return {
        "channel": channel,
        "portrait": classification["predicted_class"],
        "rendered_prompt": rendered_prompt,
        "interest_score": parsed.get("interest_score", 0.5),
        "user_reaction_text": parsed.get("user_reaction_text", ""),
        "metrics": metrics_out,
        "raw_llm_response": raw_text,
    }
