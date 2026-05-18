"""Mock-LLM: метки склонности по Jinja-контексту (задача 1 + BRD + клиент)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader

from feature_rules import LOGIT_FN, PRODUCT_IDS, SEGMENT_PRODUCT_BIAS, _sigmoid

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"
CONFIG_PRODUCTS = ROOT / "config" / "products.yaml"
CONFIG_SEGMENTS = ROOT / "config" / "segment_profiles.yaml"

FEATURE_LABELS_RU = {
    "share_last_month": "сегмент выручки (proxy)",
    "days_from_ogrn": "возраст бизнеса (дней)",
    "smb_type_code": "тип МСП",
    "okved_major": "ОКВЭД",
    "categ_name": "категория бизнеса",
    "srvpackage_sale_uk": "пакет услуг",
    "sourceattr_ccode": "канал привлечения",
    "week_sum_transactions": "оборот за неделю",
    "week_mean_transactions": "активность переводов (proxy)",
    "accum": "накопления/стабильность",
    "zpp_num_live": "уже есть ЗПП",
    "nkop_num_live": "уже есть налоговая копилка",
    "acquiring_num_live": "уже есть эквайринг",
}


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _segment_profiles() -> dict[str, Any]:
    return _load_yaml(CONFIG_SEGMENTS)["segments"]


def _brd_feature_map() -> dict[str, str]:
    return _load_yaml(CONFIG_SEGMENTS).get("brd_feature_map", {})


def _resolve_brd_value(client: pd.Series, brd_name: str, fmap: dict[str, str]) -> Any:
    col = fmap.get(brd_name, brd_name)
    if col not in client.index:
        return None
    val = client[col]
    if brd_name == "ogrn_exist_months" and col == "days_from_ogrn":
        return int(float(val) // 30)
    if brd_name == "srvpackage_zero":
        return str(val) in ("none", "base")
    if brd_name in ("categ_name_digital",):
        return str(val) in {"digital_services", "online_ads", "software", "saas"}
    if brd_name in ("categ_name_retail",):
        return str(val) in {"fuel", "equipment", "electronics", "grocery", "food_service"}
    if brd_name in ("categ_name_tax",):
        return str(val) in {"tax_payment", "bank_operations", "misc"}
    if brd_name == "sourceattr_online":
        return str(val) in ("website", "online", "api", "mobile", "social")
    return val


def _top_features_for_product(
    client: pd.Series,
    product_id: str,
    products_cfg: dict,
    fmap: dict[str, str],
    rule_score: float,
) -> list[dict[str, Any]]:
    brd_feats = products_cfg.get(product_id, {}).get("brd_features", [])
    items: list[dict[str, Any]] = []
    for name in brd_feats[:4]:
        val = _resolve_brd_value(client, name, fmap)
        if val is None:
            continue
        direction = "increases" if rule_score >= 0.5 else "decreases"
        if hasattr(val, "item"):
            val = val.item()
        items.append(
            {
                "name": name,
                "value": val,
                "direction": direction,
                "label_ru": FEATURE_LABELS_RU.get(fmap.get(name, name), name),
            }
        )
    if not items:
        items.append(
            {
                "name": "priority_segment",
                "value": str(client.get("target", "")),
                "direction": "increases",
                "label_ru": "сегмент P*",
            }
        )
    return items[:5]


def render_propensity_prompt(
    *,
    client_id: int,
    client: pd.Series,
    product_id: str,
    segment_examples: dict[str, str] | None = None,
) -> str:
    products = _load_yaml(CONFIG_PRODUCTS)["products"]
    segments = _segment_profiles()
    segment = str(client.get("target", client.get("priority_segment", "")))
    seg_prof = segments.get(segment, {})

    env = Environment(loader=FileSystemLoader(PROMPTS_DIR), autoescape=False)
    template = env.get_template("generate_client_product_row.j2")

    return template.render(
        products=list(products.keys()),
        client_id=client_id,
        priority_segment=segment,
        segment_title=seg_prof.get("title", segment),
        segment_description=seg_prof.get("description", ""),
        anchor_products=seg_prof.get("anchor_products", []),
        segment_examples=segment_examples or {},
        smb_type_code=int(client["smb_type_code"]) if pd.notna(client.get("smb_type_code")) else 0,
        okved_major=int(client["okved_major"]) if pd.notna(client.get("okved_major")) else 0,
        categ_name=client.get("categ_name", ""),
        city=client.get("city", ""),
        days_from_ogrn=int(client.get("days_from_ogrn", 0)),
        week_sum_transactions=float(client.get("week_sum_transactions", 0)),
        srvpackage_sale_uk=client.get("srvpackage_sale_uk", ""),
        product_id=product_id,
        product_name=products[product_id]["name_ru"],
        brd_features=products[product_id].get("brd_features", []),
    )


def mock_llm_propensity_response(
    *,
    client_id: int,
    client: pd.Series,
    product_id: str,
    rule_score: float,
    rule_label: int,
) -> dict[str, Any]:
    """
    Имитация ответа LLM: согласует rule_score с портретом P* и якорными продуктами BRD.
    Не вызывает внешний API.
    """
    products_cfg = _load_yaml(CONFIG_PRODUCTS)["products"]
    fmap = _brd_feature_map()
    segment = str(client.get("target", client.get("priority_segment", "")))
    seg_prof = _segment_profiles().get(segment, {})
    anchors = set(seg_prof.get("anchor_products", []))

    logit = LOGIT_FN[product_id](client)
    logit += SEGMENT_PRODUCT_BIAS.get(segment, {}).get(product_id, 0.0)
    if product_id in anchors:
        logit += 0.45
    blended = 0.55 * rule_score + 0.45 * float(_sigmoid([logit + (0.15 if product_id in anchors else -0.05)])[0])
    score = round(min(0.99, max(0.01, blended)), 6)
    label = int(score >= 0.5)

    top_features = _top_features_for_product(client, product_id, products_cfg, fmap, score)
    seg_title = seg_prof.get("title", segment)
    categ = client.get("categ_name", "бизнес")

    segment_desc = (
        f"Клиент сегмента {segment} ({seg_title}): {categ}, "
        f"портрет онбординга согласован с датасетом P1–P8."
    )

    feat_names = ", ".join(t["label_ru"] for t in top_features[:2])
    product_name = products_cfg[product_id]["name_ru"]
    draft = (
        f"Учитывая ваш профиль ({seg_title}) и сигналы по {feat_names}, "
        f"«{product_name}» выглядит логичным следующим шагом после приоритизации онбординга. "
        f"Могу кратко показить условия подключения."
    )

    return {
        "propensity_score": score,
        "propensity_label": label,
        "top_features": top_features,
        "segment_description_ru": segment_desc,
        "sales_argument_draft_ru": draft,
        "llm_reasoning_ru": (
            f"Сегмент {segment} + якорь {product_id in anchors} + BRD-фичи; "
            f"скорректировано от rule={rule_score:.2f}."
        ),
    }


def build_segment_examples(clients: pd.DataFrame, n_per_segment: int = 1) -> dict[str, str]:
    """Краткие примеры клиентов по каждому P* для контекста LLM."""
    examples: dict[str, str] = {}
    col = "target" if "target" in clients.columns else "priority_segment"
    for seg in sorted(clients[col].astype(str).unique()):
        sub = clients[clients[col].astype(str) == seg].head(n_per_segment)
        if sub.empty:
            continue
        row = sub.iloc[0]
        examples[seg] = (
            f"smb={row.get('smb_type_code')}, okved={row.get('okved_major')}, "
            f"categ={row.get('categ_name')}, days_ogrn={row.get('days_from_ogrn')}, "
            f"pkg={row.get('srvpackage_sale_uk')}"
        )
    return examples


def parse_llm_json_from_text(text: str) -> dict[str, Any]:
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1)
    return json.loads(text)
