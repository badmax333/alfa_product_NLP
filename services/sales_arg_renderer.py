"""Рендеринг шаблона sales-аргумента для отображения промпта в UI (Tab 2)."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from config.metrics import PORTRAIT_BEHAVIORAL_PROFILES
from config.sales_arguments import INTERACTION_TYPES

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_jinja_env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), autoescape=False)

_CHANNEL_LABELS = {"digital": "Цифровой канал", "voice": "Голосовой канал"}
_ITYPE_MAP = {t["id"]: t["label"] for t in INTERACTION_TYPES}


def render_sales_arg_prompt(
    classification: dict[str, Any],
    interaction_type: str,
    client_features: dict[str, Any],
) -> str:
    portrait_id = classification["predicted_class"]
    profile = PORTRAIT_BEHAVIORAL_PROFILES.get(portrait_id, {})
    itype_meta = next((t for t in INTERACTION_TYPES if t["id"] == interaction_type), {})
    channel = itype_meta.get("channel", "digital")

    template = _jinja_env.get_template("stage1_sales_argument.j2")
    return template.render(
        portrait_id=portrait_id,
        portrait_name=profile.get("name", classification.get("class_description", "")),
        typical_behavior=profile.get("typical_behavior", ""),
        negative_triggers=profile.get("negative_triggers", ""),
        top_features=classification.get("top5_feature_importance", []),
        client_features=client_features,
        product=classification.get("recommended_product", {}),
        channel=channel,
        channel_label=_CHANNEL_LABELS.get(channel, channel),
        interaction_type=interaction_type,
        interaction_type_label=_ITYPE_MAP.get(interaction_type, interaction_type),
    )
