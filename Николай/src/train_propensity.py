#!/usr/bin/env python3
"""Обучение модели склонности LightGBM и сохранение весов."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "propensity_synthetic.csv"
MODEL_PATH = ROOT / "models" / "propensity_lgbm.pkl"
CONFIG_PATH = ROOT / "models" / "feature_config.json"

CAT_FEATURES = [
    "product_id",
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

NUM_FEATURES = [
    "okved_major",
    "okved_cnt_total",
    "okved_groups_unique_cnt",
    "okved_groups_share",
    "days_from_ogrn",
    "days_from_smb",
    "ogrn_days_end_month",
    "ogrn_days_end_quarter",
    "week_sum_transactions",
    "week_mean_transactions",
    "share_last_month",
    "share_last_3_months",
    "acquiring_num_live",
    "zpp_num_live",
    "nkop_num_live",
    "rko_num_live",
    "apin_salary_last_days",
    "apin_product_active_days",
    "xpin_start_days",
    "days_from_authperson_registration",
    "prev_managers",
    "accum",
    "impnt",
    "to_activate",
    "complexity",
    "get_scores",
]

TARGET = "propensity_label"
DROP_COLS = {"propensity_score", "label_source", "client_id", "sparkcompany_uk"}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for c in NUM_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    df = load_data()
    feature_cols = [c for c in CAT_FEATURES + NUM_FEATURES if c in df.columns]
    cat_cols = [c for c in CAT_FEATURES if c in df.columns]
    num_cols = [c for c in NUM_FEATURES if c in df.columns]

    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(cat_cols, num_cols)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)
    print(f"ROC-AUC: {roc:.4f}  PR-AUC: {pr:.4f}")

  # per-product metrics
    test_df = X_test.copy()
    test_df["y"] = y_test.values
    test_df["proba"] = proba
    print("\nROC-AUC by product:")
    for pid, g in test_df.groupby("product_id"):
        if g["y"].nunique() < 2:
            continue
        print(f"  {pid}: {roc_auc_score(g['y'], g['proba']):.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    config = {
        "cat_features": cat_cols,
        "num_features": num_cols,
        "target": TARGET,
        "metrics": {"roc_auc": roc, "pr_auc": pr},
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nModel -> {MODEL_PATH}")
    print(f"Config -> {CONFIG_PATH}")


if __name__ == "__main__":
    main()
