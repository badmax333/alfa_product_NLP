"""Загрузка скриптов и сегментов из config/brd_scripts.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
BRD_SCRIPTS_PATH = ROOT / "config" / "brd_scripts.yaml"
ONBOARDING_FEATURES_PATH = ROOT / "config" / "onboarding_features.yaml"


@lru_cache(maxsize=1)
def load_brd_scripts() -> dict[str, Any]:
    with open(BRD_SCRIPTS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_onboarding_features() -> dict[str, Any]:
    with open(ONBOARDING_FEATURES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_product_script(product_id: str) -> dict[str, Any]:
    data = load_brd_scripts()
    return data.get("products", {}).get(product_id, {})


def get_zpp_segment_hints(client: dict[str, Any]) -> list[dict[str, str]]:
    """Подбор релевантного сегмента ЗПП по простым правилам (BRD)."""
    data = load_brd_scripts()
    segments = data.get("zpp_segments", [])
    smb = int(client.get("smb_type_code", 0) or 0)
    share = float(client.get("share_last_month", 0.5) or 0.5)
    days_ogrn = int(client.get("days_from_ogrn", 9999) or 9999)
    ogrn_months = days_ogrn // 30

    matched: list[dict[str, str]] = []
    if smb == 1 and share < 0.5 and ogrn_months <= 12:
        matched.append(segments[0])
    if smb == 2:
        matched.append(segments[1])
    if smb == 1 and share >= 0.5:
        matched.append(segments[2])
    return matched or segments[:1]


def get_compliance_rules() -> list[str]:
    return load_brd_scripts().get("compliance", [])
