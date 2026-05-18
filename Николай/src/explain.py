"""Интерпретация модели: SHAP → топ-признаки для sales-аргументов."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

# Человекочитаемые подписи для ключевых полей
FEATURE_LABELS_RU: dict[str, str] = {
    "priority_segment": "Сегмент приоритета онбординга",
    "smb_type_code": "Тип клиента (1=ЮЛ, 2=ИП, 3=КФХ)",
    "okved_major": "ОКВЭД (отрасль)",
    "okved_major_wrapped": "Отраслевая группа",
    "categ_name": "Категория бизнеса",
    "days_from_ogrn": "Дней с регистрации (ОГРН)",
    "days_from_smb": "Дней в реестре МСП",
    "week_sum_transactions": "Сумма транзакций за неделю",
    "week_mean_transactions": "Средняя транзакция за неделю",
    "share_last_month": "Доля активности за месяц",
    "share_last_3_months": "Доля активности за 3 месяца",
    "srvpackage_sale_uk": "Пакет услуг",
    "sourceattr_ccode": "Канал привлечения",
    "city": "Город",
    "accum": "Накопительная/стабильность",
    "impnt": "Вовлечённость в цифру",
    "complexity": "Сложность профиля",
    "zpp_num_live": "Уже подключён ЗПП",
    "nkop_num_live": "Уже подключена налоговая копилка",
    "acquiring_num_live": "Уже подключён эквайринг",
}


@dataclass
class FeatureContribution:
    feature: str
    label_ru: str
    value: Any
    shap_value: float
    direction: str  # increases | decreases


def _transformed_names(pre: Any, cat_cols: list[str], num_cols: list[str]) -> list[str]:
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = []
        for c in cat_cols:
            names.append(c)
        for c in num_cols:
            names.append(c)
    return names


def _aggregate_shap_to_original(
    shap_row: np.ndarray,
    transformed_names: list[str],
    original_cat: list[str],
    original_num: list[str],
) -> dict[str, float]:
    """Суммируем SHAP one-hot колонок в исходные признаки."""
    agg: dict[str, float] = {c: 0.0 for c in original_cat + original_num}
    for i, name in enumerate(transformed_names):
        val = float(shap_row[i])
        matched = False
        for c in original_cat:
            if name == c or name.startswith(f"{c}_") or name.startswith(f"cat__{c}_"):
                agg[c] += val
                matched = True
                break
        if not matched:
            for c in original_num:
                if name == c or name.endswith(f"__{c}") or name == f"remainder__{c}":
                    agg[c] += val
                    matched = True
                    break
        if not matched:
            m = re.match(r"(?:cat__)?([^_]+)(?:_.*)?", name)
            if m and m.group(1) in agg:
                agg[m.group(1)] += val
    return agg


def explain_client_product(
    pipe: Pipeline,
    client: pd.Series,
    product_id: str,
    *,
    cat_features: list[str],
    num_features: list[str],
    top_k: int = 5,
    exclude_features: frozenset[str] = frozenset({"product_id"}),
) -> list[FeatureContribution]:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    row = client.copy()
    if "target" in row.index:
        row["priority_segment"] = row["target"]
    row["product_id"] = product_id

    data = {c: row[c] if c in row.index else np.nan for c in cat_features + num_features}
    X = pd.DataFrame([data])
    for c in cat_features:
        X[c] = X[c].astype(str)

    Xt = pre.transform(X)
    tnames = _transformed_names(pre, cat_features, num_features)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xt)
    if isinstance(shap_values, list):
        shap_row = shap_values[1][0]
    else:
        shap_row = shap_values[0]

    agg = _aggregate_shap_to_original(shap_row, tnames, cat_features, num_features)
    agg = {k: v for k, v in agg.items() if k not in exclude_features}

    ranked = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]

    out: list[FeatureContribution] = []
    for feat, sv in ranked:
        raw_val = data.get(feat)
        out.append(
            FeatureContribution(
                feature=feat,
                label_ru=FEATURE_LABELS_RU.get(feat, feat),
                value=raw_val,
                shap_value=round(float(sv), 6),
                direction="increases" if sv > 0 else "decreases",
            )
        )
    return out


def contributions_to_dict(items: list[FeatureContribution]) -> list[dict[str, Any]]:
    return [asdict(x) for x in items]
