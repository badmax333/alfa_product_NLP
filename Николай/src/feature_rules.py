"""Правила генерации синтетической склонности (согласованы с BRD и сегментом P1–P8)."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCT_IDS = [
    "zpp",
    "alfa_payments",
    "nachalo",
    "trade_acquiring",
    "internet_acquiring",
    "tax_jar",
    "savings",
    "accounting",
]

# Сегменты P* слегка смещают базовую склонность к «своим» продуктам (связь с классификатором 1)
SEGMENT_PRODUCT_BIAS: dict[str, dict[str, float]] = {
    "P1": {"zpp": 0.15, "accounting": 0.1},
    "P2": {"alfa_payments": 0.2, "internet_acquiring": 0.15},
    "P3": {"trade_acquiring": 0.2, "savings": 0.1},
    "P4": {"internet_acquiring": 0.2, "alfa_payments": 0.1},
    "P5": {"nachalo": 0.25, "tax_jar": 0.1},
    "P6": {"accounting": 0.2, "tax_jar": 0.15},
    "P7": {"trade_acquiring": 0.15, "zpp": 0.1},
    "P8": {"zpp": 0.1, "nachalo": 0.15},
}

RETAIL_OKVED = {47, 49, 52, 53, 55, 56}
DIGITAL_CATEG = {"digital_services", "online_ads", "software", "saas", "electronics"}
RETAIL_CATEG = {"fuel", "equipment", "electronics", "food_service", "healthcare"}
TAX_CATEG = {"tax_payment", "bank_operations", "misc"}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -12, 12)))


def _revenue_segment_proxy(row: pd.Series) -> str:
    s = row["share_last_month"]
    if s < 0.25:
        return "0-5"
    if s < 0.5:
        return "5-20"
    if s < 0.75:
        return "20-90"
    if s < 0.9:
        return "90-350"
    return "350-500"


def _client_maturity_penalty(row: pd.Series) -> float:
    """Низкая зрелость клиента → ниже уверенность склонности (summary, ~30 дней)."""
    days = float(row["days_from_ogrn"])
    if days < 30:
        return -0.8
    if days < 90:
        return -0.35
    return 0.0


def logit_zpp(row: pd.Series) -> float:
    seg = _revenue_segment_proxy(row)
    smb = int(row["smb_type_code"])
    okved = int(row["okved_major"]) if pd.notna(row["okved_major"]) else 0
    payment_fiz_proxy = float(row["week_mean_transactions"]) / 30.0 + float(row["prev_managers"]) * 0.05
    ogrn_young = 1.0 if row["days_from_ogrn"] <= 365 else 0.0

    score = -1.2
    if smb in (1, 2):
        score += 0.6
    if seg in ("0-5", "5-20") and smb == 1 and ogrn_young:
        score += 1.1  # молодые ЮЛ с низкой выручкой (BRD пример)
    if smb == 2 and payment_fiz_proxy > 0.35:
        score += 0.9  # ИП с выплатами физлицам
    if seg in ("20-90", "90-350", "350-500") and float(row.get("week_sum_transactions", 0)) > 50000:
        score += 0.5
    if int(row.get("zpp_num_live", 0)) > 0:
        score -= 1.5  # уже подключён
    if okved in RETAIL_OKVED:
        score += 0.2
    return score


def logit_alfa_payments(row: pd.Series) -> float:
    pkg = str(row["srvpackage_sale_uk"])
    turnover = float(row["week_sum_transactions"])
    deb_fl_proxy = float(row["week_mean_transactions"]) * float(row["share_last_3_months"])
    score = -0.8
    if pkg not in ("none", "base"):
        score += 1.0
    if turnover > 80000:
        score += 0.9
    if deb_fl_proxy > 2:
        score += 0.6
    if float(row["abm_main_screen_proxy"] if "abm_main_screen_proxy" in row else row.get("impnt", 0)) > 0.5:
        score += 0.3
    if pkg == "none" and turnover < 10000:
        score -= 0.7
    return score


def logit_nachalo(row: pd.Series) -> float:
    pkg = str(row["srvpackage_sale_uk"])
    commission_proxy = float(row["complexity"]) * float(row["week_sum_transactions"]) / 1e6
    score = -0.5
    if pkg in ("none", "base"):
        score += 1.1
    if commission_proxy > 0.02:
        score += 0.7
    if float(row["share_last_month"]) < 0.4:
        score += 0.4
    if pkg == "premium":
        score -= 1.2
    return score


def logit_trade_acquiring(row: pd.Series) -> float:
    okved = int(row["okved_major"]) if pd.notna(row["okved_major"]) else 0
    categ = str(row["categ_name"])
    score = -1.0
    if okved in RETAIL_OKVED or categ in RETAIL_CATEG:
        score += 1.2
    if float(row["week_sum_transactions"]) > 40000:
        score += 0.5
    if int(row.get("acquiring_num_live", 0)) > 0:
        score -= 1.3
    return score


def logit_internet_acquiring(row: pd.Series) -> float:
    categ = str(row["categ_name"])
    source = str(row["sourceattr_ccode"])
    score = -0.9
    if categ in DIGITAL_CATEG:
        score += 1.1
    if source in ("website", "online", "api", "mobile", "social"):
        score += 0.6
    if float(row["impnt"]) > 0.6:
        score += 0.3
    return score


def logit_tax_jar(row: pd.Series) -> float:
    smb = int(row["smb_type_code"])
    categ = str(row["categ_name"])
    score = -0.7
    if smb == 2:
        score += 0.9
    if categ in TAX_CATEG:
        score += 0.7
    if int(row.get("nkop_num_live", 0)) > 0:
        score -= 1.4
    if float(row["days_from_ogrn"]) < 180:
        score += 0.3
    return score


def logit_savings(row: pd.Series) -> float:
    score = -0.6
    if float(row["accum"]) > 0.4:
        score += 0.8
    if float(row["share_last_3_months"]) > 0.6:
        score += 0.5
    if float(row["week_sum_transactions"]) > 30000:
        score += 0.4
    return score


def logit_accounting(row: pd.Series) -> float:
    smb = int(row["smb_type_code"])
    score = -0.8
    if smb == 2:
        score += 1.0
    if float(row["days_from_ogrn"]) < 730:
        score += 0.5
    if float(row["complexity"]) < 0.5:
        score += 0.4
    if str(row["categ_name"]) in ("tax_payment", "bank_operations", "misc"):
        score += 0.3
    return score


LOGIT_FN = {
    "zpp": logit_zpp,
    "alfa_payments": logit_alfa_payments,
    "nachalo": logit_nachalo,
    "trade_acquiring": logit_trade_acquiring,
    "internet_acquiring": logit_internet_acquiring,
    "tax_jar": logit_tax_jar,
    "savings": logit_savings,
    "accounting": logit_accounting,
}


def compute_propensity_scores(
    clients: pd.DataFrame,
    *,
    rng: np.random.Generator,
    noise_scale: float = 0.35,
) -> pd.DataFrame:
    rows: list[dict] = []

    for client_id, row in clients.iterrows():
        segment = str(row["target"])
        maturity = _client_maturity_penalty(row)

        for product_id in PRODUCT_IDS:
            logit = LOGIT_FN[product_id](row)
            logit += maturity
            logit += SEGMENT_PRODUCT_BIAS.get(segment, {}).get(product_id, 0.0)
            logit += rng.normal(0, noise_scale)

            score = float(_sigmoid(np.array([logit]))[0])
            label = int(score >= 0.5)

            rec = {
                "client_id": int(client_id),
                "priority_segment": segment,
                "product_id": product_id,
                "propensity_score": round(score, 6),
                "propensity_label": label,
                "label_source": "rule+noise",
            }
            for col in clients.columns:
                if col != "target":
                    rec[col] = row[col]
                else:
                    rec["priority_segment"] = segment
            rows.append(rec)

    return pd.DataFrame(rows)
