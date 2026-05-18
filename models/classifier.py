"""Инференс CatBoost-классификатора ступени 1 + SHAP top-5."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from config.stage1 import (
    CAT_FEATURES,
    CLASS_DESCRIPTIONS,
    DEFAULT_FEATURES,
    FEATURE_COLS,
    FEATURE_DESCRIPTIONS,
    MODEL_PATH,
    RECOMMENDED_PRODUCTS,
)


def _cat_indices() -> list[int]:
    return [i for i, c in enumerate(FEATURE_COLS) if c in CAT_FEATURES]


@lru_cache(maxsize=1)
def load_model() -> CatBoostClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def merge_features(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Собирает полный вектор признаков: база + переопределения из формы."""
    features = dict(DEFAULT_FEATURES)
    if overrides:
        for key, value in overrides.items():
            if key in FEATURE_COLS and value is not None and value != "":
                features[key] = value
    return features


def prepare_sample(raw: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for col in FEATURE_COLS:
        val = raw.get(col, np.nan)
        if col in CAT_FEATURES:
            row[col] = str(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else "nan"
        else:
            try:
                row[col] = float(val)
            except (TypeError, ValueError):
                row[col] = np.nan
    return pd.DataFrame([row], columns=FEATURE_COLS)


def _top5_shap(
    model: CatBoostClassifier,
    sample_df: pd.DataFrame,
    cat_indices: list[int],
    predicted_class: str,
    classes: np.ndarray,
) -> list[dict[str, Any]]:
    pool = Pool(sample_df, cat_features=cat_indices)
    shap_vals = model.get_feature_importance(
        data=pool,
        type="ShapValues",
        shap_calc_type="Regular",
    )
    class_idx = list(classes).index(predicted_class)
    shap_sample = shap_vals[0, :-1, class_idx]

    ranked = sorted(enumerate(shap_sample), key=lambda x: abs(x[1]), reverse=True)[:5]
    result = []
    for rank, (feat_idx, shap_val) in enumerate(ranked, 1):
        feat_name = FEATURE_COLS[feat_idx]
        result.append(
            {
                "rank": rank,
                "feature": feat_name,
                "value": sample_df.iloc[0][feat_name],
                "shap": round(float(shap_val), 5),
                "direction": "▲ в пользу сегмента" if shap_val > 0 else "▼ против сегмента",
                "description": FEATURE_DESCRIPTIONS.get(feat_name, feat_name),
            }
        )
    return result


def predict(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Предсказание сегмента и top-5 SHAP для демо-клиента."""
    raw = merge_features(overrides)
    sample_df = prepare_sample(raw)
    model = load_model()
    cat_indices = _cat_indices()
    pool = Pool(sample_df, cat_features=cat_indices)

    proba = model.predict_proba(pool)[0]
    pred_idx = int(np.argmax(proba))
    pred_cls = str(model.classes_[pred_idx])
    top5 = _top5_shap(model, sample_df, cat_indices, pred_cls, model.classes_)

    product = RECOMMENDED_PRODUCTS.get(pred_cls, {"ame": None, "name": "—"})

    return {
        "predicted_class": pred_cls,
        "class_description": CLASS_DESCRIPTIONS.get(pred_cls, pred_cls),
        "confidence": round(float(proba[pred_idx]), 4),
        "probabilities": {
            str(cls): round(float(p), 4) for cls, p in zip(model.classes_, proba)
        },
        "recommended_product": product,
        "top5_feature_importance": top5,
        "input_features": {k: raw[k] for k in FEATURE_COLS},
    }
