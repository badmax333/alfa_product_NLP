"""Согласованность P1–P8 ↔ склонность к якорным продуктам."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_anchor_map() -> dict[str, list[str]]:
    with open(ROOT / "config" / "segment_profiles.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {k: v.get("anchor_products", []) for k, v in data["segments"].items()}


def enforce_anchor_boost(df: pd.DataFrame, boost: float = 0.08) -> pd.DataFrame:
    """Слегка поднимает score для якорных продуктов сегмента (правдоподобность)."""
    anchors = load_anchor_map()
    out = df.copy()
    for idx, row in out.iterrows():
        seg = str(row["priority_segment"])
        pid = str(row["product_id"])
        if pid in anchors.get(seg, []):
            new_score = min(0.995, float(row["propensity_score"]) + boost)
            out.at[idx, "propensity_score"] = round(new_score, 6)
            out.at[idx, "propensity_label"] = int(new_score >= 0.5)
    return out


def consistency_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Доля клиентов, у которых топ-1 продукт по score ∈ anchor_products сегмента.
    """
    anchors = load_anchor_map()
    hits = 0
    total = 0
    by_segment: dict[str, dict[str, float]] = {}

    for cid, grp in df.groupby("client_id"):
        seg = str(grp["priority_segment"].iloc[0])
        top_pid = grp.sort_values("propensity_score", ascending=False).iloc[0]["product_id"]
        anchor_list = anchors.get(seg, [])
        ok = str(top_pid) in anchor_list
        hits += int(ok)
        total += 1
        if seg not in by_segment:
            by_segment[seg] = {"hit": 0, "n": 0}
        by_segment[seg]["hit"] += int(ok)
        by_segment[seg]["n"] += 1

    rate = hits / total if total else 0.0
    per_seg = {s: round(v["hit"] / v["n"], 3) for s, v in by_segment.items() if v["n"]}

    return {
        "anchor_top1_rate": round(rate, 4),
        "clients_evaluated": total,
        "per_segment": per_seg,
    }
