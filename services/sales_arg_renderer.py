"""Рендеринг шаблонов sales-аргументов через Jinja2 (Stage 1 и Stage 2)."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from config.metrics import PORTRAIT_BEHAVIORAL_PROFILES
from config.sales_arguments import INTERACTION_TYPES

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_jinja_env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), autoescape=False)

_CHANNEL_LABELS = {"digital": "Цифровой канал", "voice": "Голосовой канал"}
_ITYPE_MAP = {t["id"]: t["label"] for t in INTERACTION_TYPES}


def _base_render_context(
    classification: dict[str, Any],
    interaction_type: str,
    client_features: dict[str, Any],
) -> dict[str, Any]:
    portrait_id = classification["predicted_class"]
    profile = PORTRAIT_BEHAVIORAL_PROFILES.get(portrait_id, {})
    itype_meta = next((t for t in INTERACTION_TYPES if t["id"] == interaction_type), {})
    channel = itype_meta.get("channel", "digital")
    return {
        "portrait_id": portrait_id,
        "portrait_name": profile.get(
            "name", classification.get("class_description", "")
        ),
        "typical_behavior": profile.get("typical_behavior", ""),
        "negative_triggers": profile.get("negative_triggers", ""),
        "client_features": client_features,
        "channel": channel,
        "channel_label": _CHANNEL_LABELS.get(channel, channel),
        "interaction_type": interaction_type,
        "interaction_type_label": _ITYPE_MAP.get(interaction_type, interaction_type),
    }


def render_sales_arg_prompt(
    classification: dict[str, Any],
    interaction_type: str,
    client_features: dict[str, Any],
) -> str:
    ctx = _base_render_context(classification, interaction_type, client_features)
    ctx["top_features"] = classification.get("top5_feature_importance", [])
    ctx["product"] = classification.get("recommended_product", {})
    return _jinja_env.get_template("stage1_sales_argument.j2").render(**ctx)


def render_stage2_sales_arg_prompt(
    classification: dict[str, Any],
    interaction_type: str,
    client_features: dict[str, Any],
    propensity_product: dict[str, Any],
    stage1_argument: dict[str, Any] | None = None,
    stage1_metrics: dict[str, Any] | None = None,
) -> str:
    ctx = _base_render_context(classification, interaction_type, client_features)
    ctx.update(
        product_id=propensity_product.get("product_id", ""),
        product_name=propensity_product.get("product_name", ""),
        product_ame=propensity_product.get("product_ame"),
        product_description=propensity_product.get("description", ""),
        top_factors=propensity_product.get("top_factors", []),
        stage1_argument=stage1_argument,
        stage1_metrics=stage1_metrics,
    )
    return _jinja_env.get_template("stage2_sales_argument.j2").render(**ctx)
