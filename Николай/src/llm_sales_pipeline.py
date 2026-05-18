"""Рендер Jinja-промптов и генерация sales-аргументов (mock / OpenAI-compatible API)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader

from brd_loader import get_compliance_rules, get_product_script, get_zpp_segment_hints
from explain import FeatureContribution

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"

SMB_LABELS = {1: "Юридическое лицо", 2: "ИП", 3: "Глава КФХ"}


def _jinja_env() -> Environment:
    return Environment(loader=FileSystemLoader(PROMPTS_DIR), autoescape=False)


def _load_products() -> dict[str, Any]:
    with open(ROOT / "config" / "products.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["products"]


def _brd_context_for_product(product_id: str, client: pd.Series) -> dict[str, Any]:
    script = get_product_script(product_id)
    client_dict = client.to_dict()
    ctx: dict[str, Any] = {
        "brd_section": script.get("section", ""),
        "discovery_questions": script.get("discovery_questions", []),
        "brd_opener": script.get("opener", ""),
        "brd_benefits": script.get("benefits", []),
        "brd_closing": script.get("closing", ""),
        "brd_script_example": script.get("script_variant2_example", ""),
        "zpp_segment_examples": [],
    }
    if product_id == "zpp":
        ctx["zpp_segment_examples"] = get_zpp_segment_hints(client_dict)
    return ctx


def render_sales_prompt(
    *,
    client_id: int,
    client: pd.Series,
    product_id: str,
    product_name: str,
    propensity_score: float,
    top_features: list[FeatureContribution],
) -> str:
    products = _load_products()
    product_cfg = products.get(product_id, {})
    brd = _brd_context_for_product(product_id, client)
    template = _jinja_env().get_template("sales_argument_final.j2")
    smb = int(client["smb_type_code"]) if pd.notna(client.get("smb_type_code")) else 0

    return template.render(
        products=list(products.keys()),
        compliance_rules=get_compliance_rules(),
        client_id=client_id,
        priority_segment=str(client.get("target", client.get("priority_segment", ""))),
        smb_type_label=SMB_LABELS.get(smb, str(smb)),
        city=client.get("city", ""),
        categ_name=client.get("categ_name", ""),
        okved_major=client.get("okved_major", ""),
        days_from_ogrn=int(client.get("days_from_ogrn", 0)),
        srvpackage_sale_uk=client.get("srvpackage_sale_uk", ""),
        onboarding_scenario_id=product_cfg.get("onboarding_scenario_id"),
        product_id=product_id,
        product_name=product_name,
        propensity_score=propensity_score,
        top_features=top_features,
        brd_features=product_cfg.get("brd_features", []),
        **brd,
    )


def _mock_response(
    client: pd.Series,
    product_id: str,
    product_name: str,
    propensity_score: float,
    top_features: list[FeatureContribution],
) -> dict[str, Any]:
    """Черновик в стиле BRD-скрипта (без API)."""
    script = get_product_script(product_id)
    categ = str(client.get("categ_name", "бизнес"))
    segment = str(client.get("target", ""))
    opener = (script.get("opener") or "").strip().replace("Имя,", "").strip()
    benefits = script.get("benefits") or []
    questions = script.get("discovery_questions") or []
    feat_hint = top_features[0].label_ru if top_features else "ваш профиль"

    segment_desc = (
        f"Клиент {segment}, {SMB_LABELS.get(int(client.get('smb_type_code', 0) or 0), 'МСБ')}: "
        f"{categ}; ключевой сигнал модели — {feat_hint}."
    )

    b0 = benefits[0] if benefits else "удобное решение для вашего этапа"
    b1 = benefits[1] if len(benefits) > 1 else "экономия времени в операциях"

    if opener:
        argument = f"{opener} {b0}. {b1}. Могу коротко показать подключение — займёт несколько минут."
    else:
        argument = (
            f"Для вашего {categ} логично обсудить «{product_name}»: {b0}. "
            f"{b1}. Готов пройти с вами условия в интернет-банке."
        )

    why = (
        f"Склонность {propensity_score:.0%} и сегмент {segment} согласованы с BRD-скриптом «{script.get('section', product_id)}»."
    )

    zpp_extra = {}
    if product_id == "zpp":
        segs = get_zpp_segment_hints(client.to_dict())
        if segs:
            why += f" Близкий сегмент BRD: «{segs[0]['segment_name']}»."

    return {
        "segment_description_ru": segment_desc,
        "sales_argument_ru": argument,
        "why_it_fits_ru": why,
        "thesis_bullets": benefits[:3] or ["Преимущество для МСБ", "Быстрый старт", "Поддержка банка"],
        "discovery_question_suggested": questions[0] if questions else None,
        "brd_section": script.get("section"),
        "generator": "mock+brd",
        **zpp_extra,
    }


def _parse_json_from_llm(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1)
    return json.loads(text)


def call_llm(prompt: str, *, model: str | None = None) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; use --mock")

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai") from e

    base_url = os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Ты помощник банка МСБ. Отвечай только валидным JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    content = resp.choices[0].message.content or ""
    data = _parse_json_from_llm(content)
    data["generator"] = "llm"
    return data


def generate_sales_argument(
    *,
    client_id: int,
    client: pd.Series,
    product_id: str,
    product_name: str,
    propensity_score: float,
    top_features: list[FeatureContribution],
    use_mock: bool = True,
) -> dict[str, Any]:
    if use_mock:
        return _mock_response(client, product_id, product_name, propensity_score, top_features)

    prompt = render_sales_prompt(
        client_id=client_id,
        client=client,
        product_id=product_id,
        product_name=product_name,
        propensity_score=propensity_score,
        top_features=top_features,
    )
    return call_llm(prompt)
