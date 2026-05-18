#!/usr/bin/env python3
"""
Генерация синтетического датасета склонности.

Режимы:
  rule        — только бизнес-правила + шум (быстро)
  hybrid      — правила → mock-LLM (Jinja + BRD + P*) → согласованность (по умолчанию)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_rules import compute_propensity_scores
from propensity_llm_mock import (
    build_segment_examples,
    mock_llm_propensity_response,
    render_propensity_prompt,
)
from segment_consistency import consistency_metrics, enforce_anchor_boost

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "alfa_onboarding_dataset_5000.csv"
DEFAULT_OUT = ROOT / "data" / "propensity_synthetic.csv"
PROMPTS_DIR = ROOT / "data" / "llm_prompts_sample.jsonl"
METRICS_PATH = ROOT / "data" / "consistency_report.json"


def apply_hybrid_labels(
    clients: pd.DataFrame,
    rule_df: pd.DataFrame,
    *,
    save_prompts: int = 20,
) -> pd.DataFrame:
    segment_examples = build_segment_examples(clients)
    rows: list[dict] = []
    prompts_saved = 0

    for _, r in rule_df.iterrows():
        cid = int(r["client_id"])
        pid = str(r["product_id"])
        client = clients.iloc[cid]

        llm = mock_llm_propensity_response(
            client_id=cid,
            client=client,
            product_id=pid,
            rule_score=float(r["propensity_score"]),
            rule_label=int(r["propensity_label"]),
        )

        rec = r.to_dict()
        rec["propensity_score"] = llm["propensity_score"]
        rec["propensity_label"] = llm["propensity_label"]
        rec["label_source"] = "hybrid_rule+mock_llm"
        rec["segment_description_ru"] = llm["segment_description_ru"]
        rec["sales_argument_draft_ru"] = llm["sales_argument_draft_ru"]
        rec["llm_reasoning_ru"] = llm["llm_reasoning_ru"]
        rec["top_features_json"] = json.dumps(llm["top_features"], ensure_ascii=False)

        if prompts_saved < save_prompts:
            rec["_prompt_preview"] = render_propensity_prompt(
                client_id=cid,
                client=client,
                product_id=pid,
                segment_examples=segment_examples,
            )
            prompts_saved += 1

        rows.append(rec)

    out = pd.DataFrame(rows)
    if "_prompt_preview" in out.columns:
        out.drop(columns=["_prompt_preview"], inplace=True)

    return out


def save_prompt_samples(clients: pd.DataFrame, n: int = 10) -> None:
    segment_examples = build_segment_examples(clients)
    from feature_rules import PRODUCT_IDS

    PROMPTS_DIR.parent.mkdir(parents=True, exist_ok=True)
    with open(PROMPTS_DIR, "w", encoding="utf-8") as f:
        for cid in range(min(n, len(clients))):
            client = clients.iloc[cid]
            for pid in PRODUCT_IDS[:2]:
                prompt = render_propensity_prompt(
                    client_id=cid,
                    client=client,
                    product_id=pid,
                    segment_examples=segment_examples,
                )
                f.write(
                    json.dumps(
                        {"client_id": cid, "product_id": pid, "prompt": prompt},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate propensity synthetic dataset")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["rule", "hybrid"],
        default="hybrid",
        help="hybrid = rules + mock LLM + segment consistency",
    )
    args = parser.parse_args()

    clients = pd.read_csv(args.source).reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    rule_df = compute_propensity_scores(clients, rng=rng)

    if args.mode == "hybrid":
        long_df = apply_hybrid_labels(clients, rule_df)
        long_df = enforce_anchor_boost(long_df)
        save_prompt_samples(clients, n=5)
    else:
        long_df = rule_df

    metrics = consistency_metrics(long_df)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.output, index=False)

    n_pos = long_df["propensity_label"].sum()
    print(f"Mode: {args.mode}")
    print(f"Saved {len(long_df)} rows -> {args.output}")
    print(f"Positive rate: {n_pos / len(long_df):.3f}")
    print(f"Anchor top-1 consistency: {metrics['anchor_top1_rate']:.1%}")
    print(f"Consistency report -> {METRICS_PATH}")
    print(long_df.groupby("product_id")["propensity_label"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
