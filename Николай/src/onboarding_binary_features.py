"""Бинарные фичи онбординга (0/1): вход в АБМ/приложение, карта, кэшбэк.

Синтетика: генерируются из прокси + шум (как в проде «вошёл / не вошёл»).
Скоринг: если колонок нет во входном CSV — восстанавливаются тем же правилом без шума.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BINARY_FEATURE_COLS = [
    "abm_entered",
    "mobile_app_entered",
    "plastic_card_issued",
    "cashback_selected",
]


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def enrich_clients(
    clients: pd.DataFrame,
    *,
    rng: np.random.Generator | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Добавляет бинарные колонки, если их ещё нет (или force=True).
    rng задан — лёгкий шум для синтетики; None — детерминированные пороги для скоринга.
    """
    out = clients.copy()
    if not force and all(c in out.columns for c in BINARY_FEATURE_COLS):
        for c in BINARY_FEATURE_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).clip(0, 1).astype(int)
        return out

    impnt = _col(out, "impnt")
    apin = _col(out, "apin_product_active_days")
    days_ogrn = _col(out, "days_from_ogrn", default=999)
    week_sum = _col(out, "week_sum_transactions")
    week_mean = _col(out, "week_mean_transactions")
    accum = _col(out, "accum")

    source = out["sourceattr_ccode"].astype(str) if "sourceattr_ccode" in out.columns else pd.Series("", index=out.index)
    online_source = source.isin(["website", "online", "mobile", "social", "api"]).astype(float)

    digital = (
        0.52 * impnt
        + 0.22 * (apin / 450).clip(0, 1)
        + 0.12 * (days_ogrn < 365).astype(float)
        + 0.08 * online_source
    )
    if rng is not None:
        digital = digital + rng.normal(0, 0.1, len(out))

    out["abm_entered"] = (digital >= 0.40).astype(int)

    # Мобильное приложение: чаще при digital-канале; без АБМ — нет приложения
    mobile_score = digital + 0.15 * source.isin(["mobile", "social"]).astype(float)
    if rng is not None:
        mobile_score = mobile_score + rng.normal(0, 0.08, len(out))
    mobile_raw = (mobile_score >= 0.36).astype(int)
    out["mobile_app_entered"] = ((out["abm_entered"] == 1) & (mobile_raw == 1)).astype(int)

    card_score = (
        0.28 * (week_sum > 12_000).astype(float)
        + 0.22 * (week_mean > 3).astype(float)
        + 0.18 * impnt
        + 0.12 * (days_ogrn > 90).astype(float)
    )
    if rng is not None:
        card_score = card_score + rng.normal(0, 0.12, len(out))
    out["plastic_card_issued"] = (card_score >= 0.38).astype(int)

    cashback_score = accum + 0.25 * out["plastic_card_issued"]
    if rng is not None:
        cashback_score = cashback_score + rng.normal(0, 0.1, len(out))
    out["cashback_selected"] = (
        (out["plastic_card_issued"] == 1) & (cashback_score >= 0.42)
    ).astype(int)

    for c in BINARY_FEATURE_COLS:
        out[c] = out[c].astype(int)

    return out
