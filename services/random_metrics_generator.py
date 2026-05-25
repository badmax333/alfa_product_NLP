"""Генерация случайных метрик взаимодействия (без вызова LLM)."""

import random
from typing import Any

from config.metrics import PORTRAIT_BEHAVIORAL_PROFILES, get_metrics_for_channel

_AFFINITY_SCORE = {
    "very_low": 0.25,
    "low": 0.45,
    "medium": 0.65,
    "high": 0.82,
    "very_high": 0.93,
}

_REACTION_TEMPLATES = {
    "high": [
        "Клиент проявил явный интерес к предложению: изучил условия и перешёл к оформлению.",
        "Реакция позитивная — клиент вовлечён и готов к целевому действию.",
        "Предложение попало в точку: клиент сразу начал изучать детали продукта.",
    ],
    "medium": [
        "Клиент заметил предложение, но не спешит с решением — вернётся позже.",
        "Нейтральная реакция: интерес есть, но активного действия не последовало.",
        "Клиент посмотрел предложение, не отвергнул, но отложил решение на потом.",
    ],
    "low": [
        "Клиент практически не отреагировал на предложение — прошёл мимо.",
        "Предложение не вызвало интереса: низкая вовлечённость, нет целевых действий.",
        "Клиент проигнорировал предложение или закрыл его сразу.",
    ],
}


def _b(p: float) -> int:
    return 1 if random.random() < p else 0


def _generate_digital_values(interest: float, affinity: float) -> dict[str, Any]:
    banner_shown = _b(0.95)
    banner_visible_sec = random.randint(3, 60) if banner_shown else 0
    push_delivered = _b(0.9)
    push_opened = _b(0.30 * affinity) if push_delivered else 0

    banner_clicked = _b(0.12 * interest * affinity) if banner_shown else 0
    product_page_visited = _b(0.75) if banner_clicked else _b(0.04 * interest)
    time_on_page = random.randint(20, 300) if product_page_visited else 0
    scroll_depth = round(random.uniform(20, 100), 1) if product_page_visited else 0.0
    remind_later = _b(0.12 * interest) if product_page_visited and not banner_clicked else 0
    repeated_views = random.randint(0, 3) if product_page_visited else 0

    application_started = _b(0.35 * interest) if product_page_visited else 0
    application_completed = _b(0.65) if application_started else 0
    time_to_action = round(random.uniform(5, 2880), 1) if application_started else 0.0

    product_activated = _b(0.85) if application_completed else 0
    first_transaction = _b(0.70) if product_activated else 0
    days_to_first_use = random.randint(0, 14) if product_activated else 0

    banner_dismissed = _b(0.20) if banner_shown and not banner_clicked else 0
    push_unsubscribed = _b(max(0.0, 0.04 * (1.0 - interest))) if push_delivered else 0
    complaint_filed = _b(0.015) if banner_shown else 0
    negative_rating = max(1, min(5, round(interest * 3.5 + 1.5 + random.gauss(0, 0.4))))

    return {
        "banner_shown": banner_shown,
        "banner_visible_sec": banner_visible_sec,
        "push_delivered": push_delivered,
        "push_opened": push_opened,
        "banner_clicked": banner_clicked,
        "product_page_visited": product_page_visited,
        "time_on_product_page_sec": time_on_page,
        "scroll_depth_pct": scroll_depth,
        "remind_later_clicked": remind_later,
        "repeated_view_count": repeated_views,
        "application_started": application_started,
        "application_completed": application_completed,
        "time_to_action_min": time_to_action,
        "product_activated": product_activated,
        "first_transaction_done": first_transaction,
        "days_to_first_use": days_to_first_use,
        "banner_dismissed": banner_dismissed,
        "push_unsubscribed": push_unsubscribed,
        "complaint_filed": complaint_filed,
        "negative_rating": negative_rating,
    }


def _generate_voice_values(interest: float, affinity: float) -> dict[str, Any]:
    call_connected = _b(affinity)
    call_duration_sec = random.randint(30, 600) if call_connected else 0
    reached_block = _b(0.65) if call_connected and call_duration_sec > 60 else 0
    didnt_hangup = _b(0.5 + 0.4 * interest) if reached_block else 0

    positive = _b(0.45 * interest) if didnt_hangup else 0
    clarifying_q = random.randint(0, 4) if positive else 0
    requested_link = _b(0.28 * interest) if positive else 0
    agreed_callback = _b(0.18 * interest) if positive else 0

    verbal_agreement = _b(0.35 * interest) if positive else 0
    followed_link = _b(0.55) if verbal_agreement else 0

    product_connected = _b(0.65) if verbal_agreement else 0
    first_op = _b(0.72) if product_connected else 0

    negative = _b(max(0.0, 0.08 * (1.0 - interest))) if call_connected else 0
    complaint = _b(0.025) if negative else 0
    no_more_calls = _b(0.35) if negative else 0

    return {
        "call_connected": call_connected,
        "call_duration_sec": call_duration_sec,
        "reached_argument_block": reached_block,
        "client_didnt_hangup_before_argument": didnt_hangup,
        "positive_reaction": positive,
        "clarifying_questions_count": clarifying_q,
        "requested_link_or_materials": requested_link,
        "agreed_to_callback": agreed_callback,
        "verbal_agreement": verbal_agreement,
        "followed_link_after_call": followed_link,
        "product_connected_after_call": product_connected,
        "first_operation_after_call": first_op,
        "negative_reaction_voice": negative,
        "complaint_after_call": complaint,
        "requested_no_more_calls": no_more_calls,
    }


def _reaction_text(interest: float, portrait_id: str) -> str:
    if interest >= 0.70:
        bucket = "high"
    elif interest >= 0.45:
        bucket = "medium"
    else:
        bucket = "low"
    return random.choice(_REACTION_TEMPLATES[bucket])


def generate_metrics_random(
    classification: dict[str, Any],
    sales_argument: dict[str, Any],
    channel: str,
    client_features: dict[str, Any],
) -> dict[str, Any]:
    portrait_id = classification["predicted_class"]
    profile = PORTRAIT_BEHAVIORAL_PROFILES.get(portrait_id, {})

    interest_base = profile.get("interest_base", 0.60)
    interest_score = round(min(1.0, max(0.05, interest_base + random.gauss(0, 0.08))), 2)

    affinity_key = profile.get(
        "digital_affinity" if channel == "digital" else "voice_affinity", "medium"
    )
    affinity = _AFFINITY_SCORE.get(affinity_key, 0.65)

    raw_values = (
        _generate_digital_values(interest_score, affinity)
        if channel == "digital"
        else _generate_voice_values(interest_score, affinity)
    )

    metrics_defs = get_metrics_for_channel(channel)
    metrics_out = [
        {
            "name": m["name"],
            "label": m["label"],
            "level": m["level"],
            "level_name": m["level_name"],
            "type": m["type"],
            "unit": m["unit"],
            "description": m["description"],
            "value": raw_values.get(m["name"]),
        }
        for m in metrics_defs
    ]

    return {
        "channel": channel,
        "portrait": portrait_id,
        "rendered_prompt": "[Случайная генерация — LLM не вызывался]",
        "interest_score": interest_score,
        "user_reaction_text": _reaction_text(interest_score, portrait_id),
        "metrics": metrics_out,
        "raw_llm_response": "",
    }
