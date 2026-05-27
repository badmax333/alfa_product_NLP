"""Stage2 scoring: склонность клиента к продуктам после оценки взаимодействия."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config.propensity import (
    DIGITAL_CATEGORIES,
    PRODUCT_IDS,
    PROPENSITY_FEATURE_LABELS,
    PROPENSITY_PRODUCTS,
    RETAIL_CATEGORIES,
    RETAIL_OKVED,
    SEGMENT_PRODUCT_BIAS,
    TAX_CATEGORIES,
)
from config.stage1 import DEFAULT_FEATURES, PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "propensity_lgbm.pkl"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "models" / "feature_config.json"

_model_cache: Any | None = None
_feature_config_cache: dict[str, Any] | None = None


def _to_float(features: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        value = features.get(name, default)
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(features: dict[str, Any], name: str, default: int = 0) -> int:
    return int(_to_float(features, name, float(default)))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, value))))


def _logit(probability: float) -> float:
    bounded = min(0.999999, max(0.000001, probability))
    return math.log(bounded / (1.0 - bounded))


def _load_model_artifacts() -> tuple[Any, dict[str, Any]] | None:
    """Ленивая загрузка LightGBM pipeline из артефактов модели Николая."""
    global _model_cache, _feature_config_cache

    if _model_cache is not None and _feature_config_cache is not None:
        return _model_cache, _feature_config_cache

    if not MODEL_PATH.exists() or not FEATURE_CONFIG_PATH.exists():
        return None

    _model_cache = joblib.load(MODEL_PATH)
    _feature_config_cache = json.loads(FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))
    return _model_cache, _feature_config_cache


def _revenue_segment_proxy(features: dict[str, Any]) -> str:
    share = _to_float(features, "share_last_month", 0.0)
    if share < 0.25:
        return "0-5"
    if share < 0.5:
        return "5-20"
    if share < 0.75:
        return "20-90"
    if share < 0.9:
        return "90-350"
    return "350-500"


def _client_maturity_penalty(features: dict[str, Any]) -> tuple[float, dict[str, Any] | None]:
    days = _to_float(features, "days_from_ogrn", 0.0)
    if days < 30:
        return -0.8, _factor("days_from_ogrn", days, -0.8, "молодой бизнес: меньше 30 дней")
    if days < 90:
        return -0.35, _factor("days_from_ogrn", days, -0.35, "молодой бизнес: меньше 90 дней")
    return 0.0, None


def _factor(feature: str, value: Any, impact: float, reason: str) -> dict[str, Any]:
    return {
        "feature": feature,
        "label": PROPENSITY_FEATURE_LABELS.get(feature, feature),
        "value": value,
        "impact": round(float(impact), 4),
        "direction": "increases" if impact > 0 else "decreases",
        "reason": reason,
    }


def _score_zpp(features: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    reasons = []
    score = -1.2
    smb = _to_int(features, "smb_type_code")
    okved = _to_int(features, "okved_major")
    turnover = _to_float(features, "week_sum_transactions")
    payment_fiz_proxy = _to_float(features, "week_mean_transactions") / 30.0 + _to_float(features, "prev_managers") * 0.05
    revenue_segment = _revenue_segment_proxy(features)
    is_young = _to_float(features, "days_from_ogrn") <= 365

    if smb in (1, 2):
        score += 0.6
        reasons.append(_factor("smb_type_code", smb, 0.6, "ЮЛ/ИП подходит для зарплатного проекта"))
    if revenue_segment in ("0-5", "5-20") and smb == 1 and is_young:
        score += 1.1
        reasons.append(_factor("share_last_month", features.get("share_last_month"), 1.1, "молодое ЮЛ с небольшим оборотом"))
    if smb == 2 and payment_fiz_proxy > 0.35:
        score += 0.9
        reasons.append(_factor("week_mean_transactions", features.get("week_mean_transactions"), 0.9, "есть proxy выплат физлицам"))
    if revenue_segment in ("20-90", "90-350", "350-500") and turnover > 50000:
        score += 0.5
        reasons.append(_factor("week_sum_transactions", turnover, 0.5, "достаточный оборот для регулярных выплат"))
    if _to_int(features, "zpp_num_live") > 0:
        score -= 1.5
        reasons.append(_factor("zpp_num_live", features.get("zpp_num_live"), -1.5, "зарплатный проект уже подключен"))
    if okved in RETAIL_OKVED:
        score += 0.2
        reasons.append(_factor("okved_major", okved, 0.2, "розничная отрасль часто имеет регулярный персонал"))
    return score, reasons


def _score_alfa_payments(features: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    reasons = []
    score = -0.8
    package = str(features.get("srvpackage_sale_uk", ""))
    turnover = _to_float(features, "week_sum_transactions")
    deb_fl_proxy = _to_float(features, "week_mean_transactions") * _to_float(features, "share_last_3_months")

    if package not in ("none", "base"):
        score += 1.0
        reasons.append(_factor("srvpackage_sale_uk", package, 1.0, "клиент уже использует расширенный пакет"))
    if turnover > 80000:
        score += 0.9
        reasons.append(_factor("week_sum_transactions", turnover, 0.9, "высокий недельный оборот"))
    if deb_fl_proxy > 2:
        score += 0.6
        reasons.append(_factor("week_mean_transactions", features.get("week_mean_transactions"), 0.6, "высокая расчетная активность"))
    if _to_float(features, "impnt") > 0.5:
        score += 0.3
        reasons.append(_factor("impnt", features.get("impnt"), 0.3, "хорошая цифровая вовлеченность"))
    if package == "none" and turnover < 10000:
        score -= 0.7
        reasons.append(_factor("srvpackage_sale_uk", package, -0.7, "низкая активность без пакета услуг"))
    return score, reasons


def _score_nachalo(features: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    reasons = []
    score = -0.5
    package = str(features.get("srvpackage_sale_uk", ""))
    commission_proxy = _to_float(features, "complexity") * _to_float(features, "week_sum_transactions") / 1_000_000

    if package in ("none", "base"):
        score += 1.1
        reasons.append(_factor("srvpackage_sale_uk", package, 1.1, "стартовый или отсутствующий пакет"))
    if commission_proxy > 0.02:
        score += 0.7
        reasons.append(_factor("complexity", features.get("complexity"), 0.7, "есть потенциал экономии на комиссиях"))
    if _to_float(features, "share_last_month") < 0.4:
        score += 0.4
        reasons.append(_factor("share_last_month", features.get("share_last_month"), 0.4, "ранняя стадия активности"))
    if package == "premium":
        score -= 1.2
        reasons.append(_factor("srvpackage_sale_uk", package, -1.2, "premium-пакет уже закрывает потребность"))
    return score, reasons


def _score_trade_acquiring(features: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    reasons = []
    score = -1.0
    okved = _to_int(features, "okved_major")
    category = str(features.get("categ_name", ""))
    turnover = _to_float(features, "week_sum_transactions")

    if okved in RETAIL_OKVED or category in RETAIL_CATEGORIES:
        score += 1.2
        reasons.append(_factor("categ_name", category, 1.2, "розничный/offline контекст для приема карт"))
    if turnover > 40000:
        score += 0.5
        reasons.append(_factor("week_sum_transactions", turnover, 0.5, "оборот достаточен для эквайринга"))
    if _to_int(features, "acquiring_num_live") > 0:
        score -= 1.3
        reasons.append(_factor("acquiring_num_live", features.get("acquiring_num_live"), -1.3, "эквайринг уже подключен"))
    return score, reasons


def _score_internet_acquiring(features: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    reasons = []
    score = -0.9
    category = str(features.get("categ_name", ""))
    source = str(features.get("sourceattr_ccode", ""))

    if category in DIGITAL_CATEGORIES:
        score += 1.1
        reasons.append(_factor("categ_name", category, 1.1, "digital-категория бизнеса"))
    if source in ("website", "online", "api", "mobile", "social"):
        score += 0.6
        reasons.append(_factor("sourceattr_ccode", source, 0.6, "онлайн-канал привлечения"))
    if _to_float(features, "impnt") > 0.6:
        score += 0.3
        reasons.append(_factor("impnt", features.get("impnt"), 0.3, "высокая цифровая вовлеченность"))
    return score, reasons


def _score_tax_jar(features: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    reasons = []
    score = -0.7
    smb = _to_int(features, "smb_type_code")
    category = str(features.get("categ_name", ""))

    if smb == 2:
        score += 0.9
        reasons.append(_factor("smb_type_code", smb, 0.9, "ИП часто нужен резерв под налоги"))
    if category in TAX_CATEGORIES:
        score += 0.7
        reasons.append(_factor("categ_name", category, 0.7, "налоговый или банковский контекст операций"))
    if _to_int(features, "nkop_num_live") > 0:
        score -= 1.4
        reasons.append(_factor("nkop_num_live", features.get("nkop_num_live"), -1.4, "налоговая копилка уже подключена"))
    if _to_float(features, "days_from_ogrn") < 180:
        score += 0.3
        reasons.append(_factor("days_from_ogrn", features.get("days_from_ogrn"), 0.3, "молодому бизнесу полезен налоговый резерв"))
    return score, reasons


def _score_savings(features: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    reasons = []
    score = -0.6
    if _to_float(features, "accum") > 0.4:
        score += 0.8
        reasons.append(_factor("accum", features.get("accum"), 0.8, "есть накопительный профиль"))
    if _to_float(features, "share_last_3_months") > 0.6:
        score += 0.5
        reasons.append(_factor("share_last_3_months", features.get("share_last_3_months"), 0.5, "стабильная активность за 3 месяца"))
    turnover = _to_float(features, "week_sum_transactions")
    if turnover > 30000:
        score += 0.4
        reasons.append(_factor("week_sum_transactions", turnover, 0.4, "есть свободный денежный поток"))
    return score, reasons


def _score_accounting(features: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    reasons = []
    score = -0.8
    smb = _to_int(features, "smb_type_code")
    category = str(features.get("categ_name", ""))

    if smb == 2:
        score += 1.0
        reasons.append(_factor("smb_type_code", smb, 1.0, "ИП часто нужна простая бухгалтерия"))
    if _to_float(features, "days_from_ogrn") < 730:
        score += 0.5
        reasons.append(_factor("days_from_ogrn", features.get("days_from_ogrn"), 0.5, "молодому бизнесу важна настройка учета"))
    if _to_float(features, "complexity") < 0.5:
        score += 0.4
        reasons.append(_factor("complexity", features.get("complexity"), 0.4, "профиль подходит для типового бухгалтерского сервиса"))
    if category in ("tax_payment", "bank_operations", "misc"):
        score += 0.3
        reasons.append(_factor("categ_name", category, 0.3, "есть бухгалтерско-налоговый контекст"))
    return score, reasons


_SCORERS = {
    "zpp": _score_zpp,
    "alfa_payments": _score_alfa_payments,
    "nachalo": _score_nachalo,
    "trade_acquiring": _score_trade_acquiring,
    "internet_acquiring": _score_internet_acquiring,
    "tax_jar": _score_tax_jar,
    "savings": _score_savings,
    "accounting": _score_accounting,
}


def _build_top_factors(
    features: dict[str, Any],
    product_id: str,
    priority_segment: str,
    metrics_result: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    base_logit, reasons = _SCORERS[product_id](features)

    maturity_delta, maturity_factor = _client_maturity_penalty(features)
    if maturity_factor:
        reasons.append(maturity_factor)

    segment_delta = SEGMENT_PRODUCT_BIAS.get(priority_segment, {}).get(product_id, 0.0)
    if segment_delta:
        reasons.append(_factor("priority_segment", priority_segment, segment_delta, "связь продукта с портретом клиента"))

    interaction_delta = 0.0
    if metrics_result:
        interest = _to_float(metrics_result, "interest_score", 0.5)
        interaction_delta = (interest - 0.5) * 0.35
        if abs(interaction_delta) >= 0.03:
            reasons.append(
                _factor(
                    "interest_score",
                    round(interest, 3),
                    interaction_delta,
                    "учтен результат предыдущего взаимодействия",
                )
            )

    return sorted(reasons, key=lambda item: abs(item["impact"]), reverse=True)[:5]


def _format_product_score(
    product_id: str,
    score: float,
    model_logit: float,
    top_factors: list[dict[str, Any]],
) -> dict[str, Any]:
    product = PROPENSITY_PRODUCTS[product_id]
    return {
        "product_id": product_id,
        "product_name": product["name"],
        "product_ame": product["ame"],
        "scenario_id": product["scenario_id"],
        "description": product["description"],
        "anchor": product["anchor"],
        "propensity_score": round(float(score), 6),
        "model_logit": round(float(model_logit), 6),
        "top_factors": top_factors,
    }


def _score_product_rule_based(
    features: dict[str, Any],
    product_id: str,
    priority_segment: str,
    metrics_result: dict[str, Any] | None,
) -> dict[str, Any]:
    base_logit, _ = _SCORERS[product_id](features)
    maturity_delta, _ = _client_maturity_penalty(features)
    segment_delta = SEGMENT_PRODUCT_BIAS.get(priority_segment, {}).get(product_id, 0.0)
    interaction_delta = 0.0
    if metrics_result:
        interest = _to_float(metrics_result, "interest_score", 0.5)
        interaction_delta = (interest - 0.5) * 0.35

    final_logit = base_logit + maturity_delta + segment_delta + interaction_delta
    score = round(_sigmoid(final_logit), 6)
    top_factors = _build_top_factors(features, product_id, priority_segment, metrics_result)
    return _format_product_score(product_id, score, final_logit, top_factors)


def _client_product_matrix(
    features: dict[str, Any],
    product_id: str,
    feature_config: dict[str, Any],
) -> pd.DataFrame:
    cat_features = feature_config["cat_features"]
    num_features = feature_config["num_features"]
    row = {name: features.get(name) for name in cat_features + num_features}
    row["product_id"] = product_id
    row["priority_segment"] = features["priority_segment"]
    frame = pd.DataFrame([row])
    for name in cat_features:
        if name in frame.columns:
            frame[name] = frame[name].astype(str)
    return frame


def _score_products_with_model(
    features: dict[str, Any],
    priority_segment: str,
    metrics_result: dict[str, Any] | None,
    model: Any,
    feature_config: dict[str, Any],
) -> list[dict[str, Any]]:
    scored = []
    interaction_delta = 0.0
    if metrics_result:
        interest = _to_float(metrics_result, "interest_score", 0.5)
        interaction_delta = (interest - 0.5) * 0.35

    for product_id in PRODUCT_IDS:
        frame = _client_product_matrix(features, product_id, feature_config)
        raw_score = float(model.predict_proba(frame)[0, 1])
        calibrated_logit = _logit(raw_score) + interaction_delta
        calibrated_score = _sigmoid(calibrated_logit)
        top_factors = _build_top_factors(features, product_id, priority_segment, metrics_result)
        scored.append(_format_product_score(product_id, calibrated_score, calibrated_logit, top_factors))

    return scored


def score_propensity(
    classification: dict[str, Any],
    client_features: dict[str, Any],
    metrics_result: dict[str, Any] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Вернуть top-K продуктов по склонности для клиента после взаимодействия."""
    priority_segment = classification["predicted_class"]
    features = {**DEFAULT_FEATURES, **client_features, "priority_segment": priority_segment}

    artifacts = _load_model_artifacts()
    if artifacts:
        model, feature_config = artifacts
        scored = _score_products_with_model(
            features=features,
            priority_segment=priority_segment,
            metrics_result=metrics_result,
            model=model,
            feature_config=feature_config,
        )
        model_source = "lightgbm_propensity_lgbm"
    else:
        scored = [
            _score_product_rule_based(
                features=features,
                product_id=product_id,
                priority_segment=priority_segment,
                metrics_result=metrics_result,
            )
            for product_id in PRODUCT_IDS
        ]
        model_source = "rule_based_propensity_fallback"

    scored.sort(key=lambda item: item["propensity_score"], reverse=True)

    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank

    requested_top_k = max(1, min(int(top_k), len(scored)))
    interaction_interest = None
    if metrics_result:
        interaction_interest = metrics_result.get("interest_score")

    return {
        "portrait": priority_segment,
        "portrait_label": classification.get("class_description", ""),
        "model_source": model_source,
        "interaction_interest_score": interaction_interest,
        "top_products": scored[:requested_top_k],
        "all_products": scored,
    }
