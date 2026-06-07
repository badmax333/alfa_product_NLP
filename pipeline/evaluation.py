"""
Оценка качества аргументов: персонализированные (LLM) vs обезличенные (шаблон).

Сравниваются две стратегии на ОДНОМ наборе синтетических клиентов.
Метрики взаимодействия генерируются через LLM (Mistral) в обоих случаях —
оценщик видит текст аргумента и симулирует реакцию клиента.

  Персонализированные — LLM создаёт аргумент под портрет и признаки клиента.
    Гипотеза: выше interest_score, выше конверсия, выше релевантность.

  Обезличенные — фиксированный шаблон без персонализации по портрету.
    Гипотеза: ниже по всем метрикам качества.

Метрики сравнения:
  - interest_score (S1 и S2) — основная композитная метрика LLM-оценщика
  - conversion rate — доля активированных продуктов
  - negative action rate — доля клиентов с негативной реакцией (Level 5)
  - soft interest rate — «напомнить позже» / согласился на перезвон
  - win rate — доля клиентов, где персонализированный > обезличенного
  - Cohen's d — размер эффекта между группами
  - specificity score — конкретность аргумента по тексту (цифры, ₽, длина, CTA)
  - relevance score — LLM-оценка соответствия аргумента профилю (0–10 → 0–1)

Запуск (CLI):
    python -m pipeline.evaluation            # n=20, пауза 60с, сохраняет evaluation_results.json
    python -m pipeline.evaluation --n 3 --pause 30
    python -m pipeline.evaluation --n 50 --output results/eval_50.json
"""

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

from config.sales_arguments import INTERACTION_TYPES
from pipeline.full_pipeline import full_run_single, generate_random_client_features
from services.llm import call_mistral


# ---------------------------------------------------------------------------
# Вспомогательные функции для статистики
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def _std(values: list[float]) -> float:
    return round(statistics.stdev(values), 4) if len(values) > 1 else 0.0


def _rate(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _extract(results: list[dict], key: str) -> list:
    return [r["summary"][key] for r in results]


def _cohens_d(a: list[float], b: list[float]) -> float:
    """Размер эффекта Cohen's d: насколько группы различаются относительно разброса."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_std = ((statistics.stdev(a) ** 2 + statistics.stdev(b) ** 2) / 2) ** 0.5
    return (
        round((statistics.mean(a) - statistics.mean(b)) / pooled_std, 4)
        if pooled_std > 0
        else 0.0
    )


# ---------------------------------------------------------------------------
# Метрики качества аргумента
# ---------------------------------------------------------------------------

_NEGATIVE_METRIC_NAMES = {
    "banner_dismissed",
    "complaint_filed",
    "push_unsubscribed",
    "negative_reaction_voice",
    "requested_no_more_calls",
    "complaint_after_call",
}

_SOFT_INTEREST_METRIC_NAMES = {"remind_later_clicked", "agreed_to_callback"}


def _has_negative_action(result: dict) -> bool:
    """True, если у клиента было хотя бы одно негативное действие на S1 или S2."""
    all_metrics = result.get("s1_metrics", {}).get("metrics", []) + result.get(
        "s2_metrics", {}
    ).get("metrics", [])
    return any(
        m["name"] in _NEGATIVE_METRIC_NAMES and bool(m.get("value"))
        for m in all_metrics
    )


def _has_soft_interest(result: dict) -> bool:
    """True, если клиент проявил мягкий интерес (remind_later или callback)."""
    all_metrics = result.get("s1_metrics", {}).get("metrics", []) + result.get(
        "s2_metrics", {}
    ).get("metrics", [])
    return any(
        m["name"] in _SOFT_INTEREST_METRIC_NAMES and bool(m.get("value"))
        for m in all_metrics
    )


def _specificity_score(argument: dict) -> float:
    """
    Конкретность аргумента по тексту (0–1), без вызова LLM.
      +0.35 — есть числа в тексте
      +0.25 — есть ₽ или %
      +0.20 — длина > 150 символов (достаточная детализация)
      +0.20 — CTA содержит конкретный глагол действия
    """
    text = " ".join(
        filter(
            None,
            [
                argument.get("headline", ""),
                argument.get("body", ""),
                argument.get("cta", ""),
            ],
        )
    )
    score = 0.0
    if re.search(r"\d+", text):
        score += 0.35
    if re.search(r"[₽%]", text):
        score += 0.25
    if len(text) > 150:
        score += 0.20
    cta = argument.get("cta", "").lower()
    if any(
        w in cta for w in ["подключ", "оформ", "запуст", "выведи", "попробу", "активир"]
    ):
        score += 0.20
    return round(min(score, 1.0), 3)


def _score_relevance_llm(argument: dict, client_features: dict, portrait: str) -> float:
    """
    LLM-оценка релевантности аргумента профилю клиента (0–1).
    Задаёт Mistral прямой вопрос: насколько аргумент соответствует бизнесу клиента?
    Возвращает 0.5 при ошибке (нейтральный фолбэк).
    """
    prompt = (
        "Оцени релевантность sales-аргумента данному бизнес-клиенту.\n\n"
        f"Профиль клиента:\n"
        f"- Портрет: {portrait}\n"
        f"- Тип бизнеса: {client_features.get('smb_type_code', '?')}\n"
        f"- Отрасль: {client_features.get('okved_major_wrapped', '?')}\n"
        f"- Вид деятельности: {client_features.get('main_okved', '?')}\n\n"
        f"Аргумент:\n"
        f"Заголовок: {argument.get('headline', '')}\n"
        f"Текст: {argument.get('body', '')}\n"
        f"CTA: {argument.get('cta', '')}\n\n"
        "Верни ТОЛЬКО одно целое число от 0 до 10, где:\n"
        "0 — аргумент не связан с профилем клиента\n"
        "10 — аргумент точно попадает в потребности и специфику этого бизнеса\n"
        "Число:"
    )
    try:
        result = call_mistral(prompt, temperature=0.0, max_tokens=10)
        match = re.search(r"\d+(?:\.\d+)?", result or "")
        return round(float(match.group()) / 10.0, 3) if match else 0.5
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Агрегация статистики
# ---------------------------------------------------------------------------


def _compute_stats(personalized: list[dict], generic: list[dict]) -> dict[str, Any]:
    """Считает сводную статистику по двум группам клиентов."""
    p_s1 = _extract(personalized, "s1_interest")
    p_s2 = _extract(personalized, "s2_interest")
    g_s1 = _extract(generic, "s1_interest")
    g_s2 = _extract(generic, "s2_interest")

    p_s1_conv = _extract(personalized, "s1_activated")
    p_s2_conv = _extract(personalized, "s2_activated")
    g_s1_conv = _extract(generic, "s1_activated")
    g_s2_conv = _extract(generic, "s2_activated")

    # Негативные действия и мягкий интерес
    p_neg = [_has_negative_action(r) for r in personalized]
    g_neg = [_has_negative_action(r) for r in generic]
    p_soft = [_has_soft_interest(r) for r in personalized]
    g_soft = [_has_soft_interest(r) for r in generic]

    # Win rate — на каждом клиенте персонализированный vs обезличенный
    win_s1 = sum(
        1
        for p, g in zip(personalized, generic)
        if p["summary"]["s1_interest"] > g["summary"]["s1_interest"]
    )
    win_s2 = sum(
        1
        for p, g in zip(personalized, generic)
        if p["summary"]["s2_interest"] > g["summary"]["s2_interest"]
    )
    n = len(personalized)

    # Specificity (текстовая метрика, без LLM)
    p_spec = [_specificity_score(r["s1_argument"]) for r in personalized]
    g_spec = [_specificity_score(r["s1_argument"]) for r in generic]

    # Relevance (LLM-оценка, записана в _s1_relevance при прогоне)
    p_rel = [r.get("_s1_relevance", 0.5) for r in personalized]
    g_rel = [r.get("_s1_relevance", 0.5) for r in generic]

    def portrait_breakdown(results: list[dict]) -> dict[str, dict]:
        by_portrait: dict[str, list] = {}
        for r in results:
            p = r["summary"]["portrait"]
            by_portrait.setdefault(p, []).append(r["summary"]["s1_interest"])
        return {
            p: {"n": len(v), "s1_mean": _mean(v), "s1_std": _std(v)}
            for p, v in sorted(by_portrait.items())
        }

    return {
        "n": n,
        "personalized": {
            "s1_interest": {"mean": _mean(p_s1), "std": _std(p_s1)},
            "s2_interest": {"mean": _mean(p_s2), "std": _std(p_s2)},
            "s1_to_s2_lift": round(_mean(p_s2) - _mean(p_s1), 4),
            "s1_conversion": _rate(p_s1_conv),
            "s2_conversion": _rate(p_s2_conv),
            "negative_rate": _rate(p_neg),
            "soft_interest_rate": _rate(p_soft),
            "specificity_mean": _mean(p_spec),
            "relevance_mean": _mean(p_rel),
            "by_portrait": portrait_breakdown(personalized),
        },
        "generic": {
            "s1_interest": {"mean": _mean(g_s1), "std": _std(g_s1)},
            "s2_interest": {"mean": _mean(g_s2), "std": _std(g_s2)},
            "s1_to_s2_lift": round(_mean(g_s2) - _mean(g_s1), 4),
            "s1_conversion": _rate(g_s1_conv),
            "s2_conversion": _rate(g_s2_conv),
            "negative_rate": _rate(g_neg),
            "soft_interest_rate": _rate(g_soft),
            "specificity_mean": _mean(g_spec),
            "relevance_mean": _mean(g_rel),
            "by_portrait": portrait_breakdown(generic),
        },
        "delta": {
            "s1_interest": round(_mean(p_s1) - _mean(g_s1), 4),
            "s2_interest": round(_mean(p_s2) - _mean(g_s2), 4),
            "s1_conversion": round(_rate(p_s1_conv) - _rate(g_s1_conv), 4),
            "s2_conversion": round(_rate(p_s2_conv) - _rate(g_s2_conv), 4),
            "negative_rate": round(_rate(p_neg) - _rate(g_neg), 4),
            "soft_interest_rate": round(_rate(p_soft) - _rate(g_soft), 4),
            "specificity": round(_mean(p_spec) - _mean(g_spec), 4),
            "relevance": round(_mean(p_rel) - _mean(g_rel), 4),
        },
        "win_rate": {
            "s1": round(win_s1 / n, 4) if n else 0.0,
            "s2": round(win_s2 / n, 4) if n else 0.0,
        },
        "cohens_d": {
            "s1_interest": _cohens_d(p_s1, g_s1),
            "s2_interest": _cohens_d(p_s2, g_s2),
        },
        "llm_fallback_count": sum(
            1
            for r in personalized
            if r.get("llm_errors", {}).get("s1") or r.get("llm_errors", {}).get("s2")
        ),
    }


# ---------------------------------------------------------------------------
# Запуск эксперимента
# ---------------------------------------------------------------------------


def run_experiment(
    n: int,
    shared_features: list[dict],
    shared_itypes: list[tuple[str, str]],
    client_pause: int = 60,
) -> dict[str, Any]:
    """
    Запускает n клиентов × 2 стратегии (персонализированная / обезличенная).

    Для каждого клиента:
      1. Персонализированный прогон (LLM аргументы + LLM метрики)
      2. Обезличенный прогон (шаблон + LLM метрики)
      3. LLM-оценка релевантности S1-аргументов обеих стратегий

    client_pause: пауза в секундах между клиентами (по умолчанию 60).
    """
    personalized_results: list[dict] = []
    generic_results: list[dict] = []

    for i, (features, (s1_itype, s2_itype)) in enumerate(
        zip(shared_features, shared_itypes), start=1
    ):
        print(f"  [{i:>2}/{n}]", end=" ", flush=True)

        p = full_run_single(
            client_features=features,
            s1_interaction_type=s1_itype,
            s2_interaction_type=s2_itype,
            s1_personalized=True,
            s2_personalized=True,
            metrics_method="llm",
        )
        g = full_run_single(
            client_features=features,
            s1_interaction_type=s1_itype,
            s2_interaction_type=s2_itype,
            s1_personalized=False,
            s2_personalized=False,
            metrics_method="llm",
        )

        # LLM-оценка релевантности S1-аргументов
        portrait = p["classification"]["predicted_class"]
        p["_s1_relevance"] = _score_relevance_llm(p["s1_argument"], features, portrait)
        g["_s1_relevance"] = _score_relevance_llm(g["s1_argument"], features, portrait)

        personalized_results.append(p)
        generic_results.append(g)

        ps = p["summary"]
        gs = g["summary"]
        fb = " [llm_fallback]" if ps.get("llm_fallback") else ""
        print(
            f"portrait={ps['portrait']}  "
            f"Pers: s1={ps['s1_interest']:.2f} s2={ps['s2_interest']:.2f} rel={p['_s1_relevance']:.2f}  "
            f"Generic: s1={gs['s1_interest']:.2f} s2={gs['s2_interest']:.2f} rel={g['_s1_relevance']:.2f}"
            f"{fb}"
        )

        if i < n and client_pause > 0:
            print(f"  → пауза {client_pause}с перед следующим клиентом...", flush=True)
            time.sleep(client_pause)

    return {
        "personalized": personalized_results,
        "generic": generic_results,
        "stats": _compute_stats(personalized_results, generic_results),
    }


# ---------------------------------------------------------------------------
# Вывод отчёта
# ---------------------------------------------------------------------------


def _print_report(stats: dict[str, Any]) -> None:
    n = stats["n"]
    p = stats["personalized"]
    g = stats["generic"]
    d = stats["delta"]
    wr = stats["win_rate"]
    cd = stats["cohens_d"]
    fb = stats["llm_fallback_count"]

    bar = "=" * 72
    row = "{:<34} {:>16} {:>12}  {:>6}"
    sep = "-" * 72

    print(f"\n{bar}")
    print(
        f"  РЕЗУЛЬТАТЫ  |  n={n}  |  оценка: LLM (Mistral)"
        + (f"  |  llm_fallbacks={fb}" if fb else "")
    )
    print(bar)
    print(row.format("Метрика", "Персонализированные", "Обезличенные", "Δ"))
    print(sep)

    # Основные метрики интереса
    print(
        row.format(
            "S1 interest (mean ± std)",
            f"{p['s1_interest']['mean']:.3f} ±{p['s1_interest']['std']:.3f}",
            f"{g['s1_interest']['mean']:.3f} ±{g['s1_interest']['std']:.3f}",
            f"{d['s1_interest']:+.3f}",
        )
    )
    print(
        row.format(
            "S2 interest (mean ± std)",
            f"{p['s2_interest']['mean']:.3f} ±{p['s2_interest']['std']:.3f}",
            f"{g['s2_interest']['mean']:.3f} ±{g['s2_interest']['std']:.3f}",
            f"{d['s2_interest']:+.3f}",
        )
    )
    print(
        row.format(
            "S1 → S2 interest lift",
            f"{p['s1_to_s2_lift']:+.3f}",
            f"{g['s1_to_s2_lift']:+.3f}",
            "",
        )
    )
    print(sep)

    # Конверсия
    print(
        row.format(
            "S1 conversion rate",
            f"{p['s1_conversion']:.1%}",
            f"{g['s1_conversion']:.1%}",
            f"{d['s1_conversion']:+.1%}",
        )
    )
    print(
        row.format(
            "S2 conversion rate",
            f"{p['s2_conversion']:.1%}",
            f"{g['s2_conversion']:.1%}",
            f"{d['s2_conversion']:+.1%}",
        )
    )
    print(sep)

    # Поведенческие метрики
    print(
        row.format(
            "Negative action rate",
            f"{p['negative_rate']:.1%}",
            f"{g['negative_rate']:.1%}",
            f"{d['negative_rate']:+.1%}",
        )
    )
    print(
        row.format(
            "Soft interest rate",
            f"{p['soft_interest_rate']:.1%}",
            f"{g['soft_interest_rate']:.1%}",
            f"{d['soft_interest_rate']:+.1%}",
        )
    )
    print(sep)

    # Качество аргумента
    print(
        row.format(
            "Specificity score (text)",
            f"{p['specificity_mean']:.3f}",
            f"{g['specificity_mean']:.3f}",
            f"{d['specificity']:+.3f}",
        )
    )
    print(
        row.format(
            "Relevance score (LLM 0–1)",
            f"{p['relevance_mean']:.3f}",
            f"{g['relevance_mean']:.3f}",
            f"{d['relevance']:+.3f}",
        )
    )
    print(sep)

    # Статистика сравнения
    print(
        row.format(
            "Win rate S1 (pers > generic)",
            f"{wr['s1']:.1%}",
            "—",
            "",
        )
    )
    print(
        row.format(
            "Win rate S2 (pers > generic)",
            f"{wr['s2']:.1%}",
            "—",
            "",
        )
    )
    print(
        row.format(
            "Cohen's d  (S1 interest)",
            f"{cd['s1_interest']:.3f}",
            "—",
            "",
        )
    )
    print(
        row.format(
            "Cohen's d  (S2 interest)",
            f"{cd['s2_interest']:.3f}",
            "—",
            "",
        )
    )
    print(bar)

    # Разбивка по портретам
    print("\n  S1 interest по портрету:")
    all_portraits = sorted(set(p["by_portrait"]) | set(g["by_portrait"]))
    for portrait in all_portraits:
        pb = p["by_portrait"].get(portrait, {})
        gb = g["by_portrait"].get(portrait, {})
        pm = pb.get("s1_mean", 0.0)
        gm = gb.get("s1_mean", 0.0)
        pn = pb.get("n", 0)
        print(
            f"    {portrait}: Pers={pm:.3f} (n={pn})  Generic={gm:.3f}  Δ={pm - gm:+.3f}"
        )

    # Интерпретация
    print()
    s1_d = d["s1_interest"]
    cds1 = cd["s1_interest"]
    if s1_d > 0.05 and cds1 > 0.2:
        print(
            f"  ✓ Персонализированные аргументы на {s1_d:+.3f} выше по S1 interest "
            f"(Cohen's d={cds1:.2f}).\n"
            f"    LLM-оценщик фиксирует разницу в качестве аргументов."
        )
    elif s1_d > 0.01:
        print(
            f"  ~ Небольшой положительный эффект ({s1_d:+.3f}, d={cds1:.2f}). "
            f"Увеличьте n для уверенного сигнала."
        )
    else:
        print(
            f"  ✗ Явного преимущества не обнаружено ({s1_d:+.3f}, d={cds1:.2f}). "
            f"Проверьте качество генерации или увеличьте n."
        )
    print()


# ---------------------------------------------------------------------------
# Полная оценка
# ---------------------------------------------------------------------------


def run_full_evaluation(
    n: int = 20,
    output_json: str | None = None,
    client_pause: int = 60,
) -> None:
    """
    Сравнивает персонализированные (LLM) и обезличенные (шаблон) аргументы
    на n синтетических клиентах. Метрики в обоих случаях через Mistral.

    Оценка числа вызовов Mistral для n клиентов:
      Персонализированные: n × (S1 арг + S2 арг + S1 метрики + S2 метрики) = 4n
      Обезличенные:        n × (S1 метрики + S2 метрики) = 2n
      Relevance scoring:   n × 2 = 2n
      Итого: ~8n вызовов   (n=3 → ~24 вызова)
    """
    itype_ids = [t["id"] for t in INTERACTION_TYPES]
    shared_features = [generate_random_client_features() for _ in range(n)]
    shared_itypes = [
        (random.choice(itype_ids), random.choice(itype_ids)) for _ in range(n)
    ]

    header = "=" * 72
    print(f"\n{header}")
    print(f"  ОЦЕНКА ПЕРСОНАЛИЗАЦИИ  |  n={n} клиентов × 2 стратегии")
    print(f"  ~{8 * n} вызовов Mistral API  |  обе стратегии оцениваются LLM")
    print("  Персонализированные: LLM создаёт аргумент под портрет клиента")
    print("  Обезличенные: фиксированный шаблон без персонализации")
    print(f"  Пауза между клиентами: {client_pause}с")
    print(header)
    print()

    result = run_experiment(
        n, shared_features, shared_itypes, client_pause=client_pause
    )
    _print_report(result["stats"])

    output = {
        "n": n,
        "stats": result["stats"],
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    save_path = results_dir / (output_json or "evaluation_results.json")
    Path(save_path).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Результаты сохранены → {save_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Оценка: персонализированные LLM-аргументы vs обезличенные шаблоны."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Количество синтетических клиентов (по умолчанию: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь для сохранения JSON-результатов (по умолчанию: evaluation_results.json)",
    )
    parser.add_argument(
        "--pause",
        type=int,
        default=60,
        help="Пауза в секундах между клиентами (по умолчанию: 60)",
    )
    args = parser.parse_args()
    run_full_evaluation(n=args.n, output_json=args.output, client_pause=args.pause)
