"""
Оценка качества аргументов: персонализированные (LLM) vs обезличенные (шаблон).

Сравниваются две стратегии на ОДНОМ наборе синтетических клиентов.
Метрики взаимодействия генерируются через LLM (Mistral) в обоих случаях —
оценщик видит текст аргумента и симулирует реакцию клиента.

  Персонализированные — LLM создаёт аргумент под портрет и признаки клиента.
    Гипотеза: выше interest_score, выше конверсия.

  Обезличенные — фиксированный шаблон без персонализации по портрету.
    Гипотеза: ниже interest_score — LLM фиксирует меньшую релевантность.

Если Δ (персонализированные − обезличенные) > 0 — персонализация работает:
LLM-оценщик видит разницу в качестве аргументов.

Запуск (CLI):
    python -m pipeline.evaluation            # n=20, сохраняет evaluation_results.json
    python -m pipeline.evaluation --n 10     # быстрый тест
    python -m pipeline.evaluation --n 50 --output results/eval_50.json
"""

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from config.sales_arguments import INTERACTION_TYPES
from pipeline.full_pipeline import full_run_single, generate_random_client_features


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
        "n": len(personalized),
        "personalized": {
            "s1_interest": {"mean": _mean(p_s1), "std": _std(p_s1)},
            "s2_interest": {"mean": _mean(p_s2), "std": _std(p_s2)},
            "s1_to_s2_lift": round(_mean(p_s2) - _mean(p_s1), 4),
            "s1_conversion": _rate(p_s1_conv),
            "s2_conversion": _rate(p_s2_conv),
            "by_portrait": portrait_breakdown(personalized),
        },
        "generic": {
            "s1_interest": {"mean": _mean(g_s1), "std": _std(g_s1)},
            "s2_interest": {"mean": _mean(g_s2), "std": _std(g_s2)},
            "s1_to_s2_lift": round(_mean(g_s2) - _mean(g_s1), 4),
            "s1_conversion": _rate(g_s1_conv),
            "s2_conversion": _rate(g_s2_conv),
            "by_portrait": portrait_breakdown(generic),
        },
        "delta": {
            "s1_interest": round(_mean(p_s1) - _mean(g_s1), 4),
            "s2_interest": round(_mean(p_s2) - _mean(g_s2), 4),
            "s1_conversion": round(_rate(p_s1_conv) - _rate(g_s1_conv), 4),
            "s2_conversion": round(_rate(p_s2_conv) - _rate(g_s2_conv), 4),
        },
        "llm_fallback_count": sum(
            1 for r in personalized
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
) -> dict[str, Any]:
    """
    Запускает n клиентов × 2 стратегии (персонализированная / обезличенная).

    Обе стратегии используют LLM для оценки метрик — Mistral симулирует
    реакцию клиента на каждый из аргументов. Отличие только в тексте аргумента.
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
        personalized_results.append(p)

        g = full_run_single(
            client_features=features,
            s1_interaction_type=s1_itype,
            s2_interaction_type=s2_itype,
            s1_personalized=False,
            s2_personalized=False,
            metrics_method="llm",
        )
        generic_results.append(g)

        ps = p["summary"]
        gs = g["summary"]
        fb = " [llm_fallback]" if ps.get("llm_fallback") else ""
        print(
            f"portrait={ps['portrait']}  "
            f"Pers: s1={ps['s1_interest']:.2f} s2={ps['s2_interest']:.2f}  "
            f"Generic: s1={gs['s1_interest']:.2f} s2={gs['s2_interest']:.2f}{fb}"
        )

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
    fb = stats["llm_fallback_count"]

    bar = "=" * 72
    row = "{:<34} {:>16} {:>12}  {:>6}"
    sep = "-" * 72

    print(f"\n{bar}")
    print(f"  РЕЗУЛЬТАТЫ  |  n={n}  |  оценка: LLM (Mistral)" +
          (f"  |  llm_fallbacks={fb}" if fb else ""))
    print(bar)
    print(row.format("Метрика", "Персонализированные", "Обезличенные", "Δ"))
    print(sep)
    print(row.format(
        "S1 interest (mean ± std)",
        f"{p['s1_interest']['mean']:.3f} ±{p['s1_interest']['std']:.3f}",
        f"{g['s1_interest']['mean']:.3f} ±{g['s1_interest']['std']:.3f}",
        f"{d['s1_interest']:+.3f}",
    ))
    print(row.format(
        "S2 interest (mean ± std)",
        f"{p['s2_interest']['mean']:.3f} ±{p['s2_interest']['std']:.3f}",
        f"{g['s2_interest']['mean']:.3f} ±{g['s2_interest']['std']:.3f}",
        f"{d['s2_interest']:+.3f}",
    ))
    print(row.format(
        "S1 → S2 interest lift",
        f"{p['s1_to_s2_lift']:+.3f}",
        f"{g['s1_to_s2_lift']:+.3f}",
        "",
    ))
    print(sep)
    print(row.format(
        "S1 conversion rate",
        f"{p['s1_conversion']:.1%}",
        f"{g['s1_conversion']:.1%}",
        f"{d['s1_conversion']:+.1%}",
    ))
    print(row.format(
        "S2 conversion rate",
        f"{p['s2_conversion']:.1%}",
        f"{g['s2_conversion']:.1%}",
        f"{d['s2_conversion']:+.1%}",
    ))
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
        print(f"    {portrait}: Pers={pm:.3f} (n={pn})  Generic={gm:.3f}  Δ={pm - gm:+.3f}")

    # Интерпретация
    print()
    if d["s1_interest"] > 0.05:
        print(
            f"  ✓ Персонализированные аргументы на {d['s1_interest']:+.3f} выше по S1 interest.\n"
            f"    LLM-оценщик фиксирует разницу в качестве аргументов."
        )
    elif d["s1_interest"] > 0.01:
        print(f"  ~ Небольшой положительный эффект ({d['s1_interest']:+.3f}). Увеличьте n для уверенного сигнала.")
    else:
        print(f"  ✗ Явного преимущества не обнаружено ({d['s1_interest']:+.3f}). "
              f"Проверьте качество генерации аргументов или увеличьте n.")
    print()


# ---------------------------------------------------------------------------
# Полная оценка
# ---------------------------------------------------------------------------

def run_full_evaluation(n: int = 20, output_json: str | None = None) -> None:
    """
    Сравнивает персонализированные (LLM) и обезличенные (шаблон) аргументы
    на n синтетических клиентах. Метрики взаимодействия в обоих случаях
    генерируются через Mistral — оценщик видит текст аргумента и реакцию.

    Если Δ interest_score > 0 — персонализация повышает вовлечённость:
    LLM-оценщик измеряет реальное качество аргумента.

    Оценка числа вызовов Mistral для n клиентов:
      Персонализированные: n × (S1 аргумент + S2 аргумент + S1 метрики + S2 метрики) = 4n
      Обезличенные: n × (S1 метрики + S2 метрики) = 2n  (аргументы — шаблоны, LLM не нужен)
      Итого: ~6n вызовов   (n=20 → ~120 вызовов, ≈ 2–6 мин)
    """
    itype_ids = [t["id"] for t in INTERACTION_TYPES]
    shared_features = [generate_random_client_features() for _ in range(n)]
    shared_itypes = [
        (random.choice(itype_ids), random.choice(itype_ids)) for _ in range(n)
    ]

    header = "=" * 72
    print(f"\n{header}")
    print(f"  ОЦЕНКА ПЕРСОНАЛИЗАЦИИ  |  n={n} клиентов × 2 стратегии")
    print(f"  ~{6 * n} вызовов Mistral API  |  обе стратегии оцениваются LLM")
    print(f"  Персонализированные: LLM создаёт аргумент под портрет клиента")
    print(f"  Обезличенные: фиксированный шаблон без персонализации")
    print(header)
    print()

    result = run_experiment(n, shared_features, shared_itypes)
    _print_report(result["stats"])

    output = {
        "n": n,
        "stats": result["stats"],
    }

    save_path = output_json or "evaluation_results.json"
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
        "--n", type=int, default=20,
        help="Количество синтетических клиентов (по умолчанию: 20, ~120 вызовов Mistral)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Путь для сохранения JSON-результатов (по умолчанию: evaluation_results.json)",
    )
    args = parser.parse_args()
    run_full_evaluation(n=args.n, output_json=args.output)
