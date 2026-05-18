#!/usr/bin/env python3
"""Отдельная LightGBM на каждый продукт (без product_id в признаках)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from feature_rules import PRODUCT_IDS
from train_propensity import NUM_FEATURES, TARGET, build_pipeline, load_data

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models" / "per_product"
METRICS_PATH = OUT_DIR / "metrics.json"

# Признаки без product_id
CAT_FEATURES_PP = [
    "priority_segment",
    "smb_type_code",
    "okved_major_wrapped",
    "categ_name",
    "srvpackage_sale_uk",
    "sourceattr_ccode",
    "city",
    "addrf_region_name",
    "division_name",
]


def train_one(df: pd.DataFrame, product_id: str) -> dict:
    sub = df[df["product_id"] == product_id].copy()
    cat_cols = [c for c in CAT_FEATURES_PP if c in sub.columns]
    num_cols = [c for c in NUM_FEATURES if c in sub.columns]
    feature_cols = cat_cols + num_cols

    X = sub[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    y = sub[TARGET].astype(int)

    if y.nunique() < 2:
        return {"product_id": product_id, "skipped": True, "reason": "single class"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline(cat_cols, num_cols)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{product_id}.pkl"
    joblib.dump(pipe, path)

    return {
        "product_id": product_id,
        "model_path": str(path),
        "n_train": len(X_train),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }


def main() -> None:
    df = load_data()
    results = []
    for pid in PRODUCT_IDS:
        m = train_one(df, pid)
        results.append(m)
        if m.get("skipped"):
            print(f"  {pid}: SKIP")
        else:
            print(f"  {pid}: ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f}")

    METRICS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
