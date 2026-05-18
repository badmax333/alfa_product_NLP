#!/usr/bin/env python3
"""Батч-скоринг: для каждого клиента топ-K продуктов + SHAP-признаки + промпт."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from explain import contributions_to_dict, explain_client_product
from scoring import PropensityScorer, load_onboarding_clients

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "scored_clients_top3.jsonl"
DEFAULT_CSV = ROOT / "data" / "scored_clients_top3.csv"


def build_record(
    scorer: PropensityScorer,
    client_id: int,
    client: pd.Series,
    *,
    top_k: int,
    explain_k: int,
    render_prompt: bool,
) -> list[dict]:
    from llm_sales_pipeline import render_sales_prompt

    records = []
    top_products = scorer.score_client(client, top_k=top_k)

    for ps in top_products:
        feats = explain_client_product(
            scorer.pipe,
            client,
            ps.product_id,
            cat_features=scorer.cat_features,
            num_features=scorer.num_features,
            top_k=explain_k,
        )
        rec = {
            "client_id": client_id,
            "priority_segment": str(client.get("target", client.get("priority_segment", ""))),
            "product_id": ps.product_id,
            "product_name": ps.product_name,
            "rank": ps.rank,
            "propensity_score": ps.propensity_score,
            "top_features": contributions_to_dict(feats),
        }
        if render_prompt:
            rec["sales_prompt"] = render_sales_prompt(
                client_id=client_id,
                client=client,
                product_id=ps.product_id,
                product_name=ps.product_name,
                propensity_score=ps.propensity_score,
                top_features=feats,
            )
        records.append(rec)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch propensity scoring with SHAP")
    parser.add_argument("--limit", type=int, default=100, help="Number of clients (0 = all)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--explain-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--prompts", action="store_true", help="Include rendered Jinja prompts")
    parser.add_argument("--source", type=Path, default=None)
    args = parser.parse_args()

    clients = load_onboarding_clients(args.source)
    if args.limit > 0:
        clients = clients.head(args.limit)

    scorer = PropensityScorer()
    all_records: list[dict] = []

    for idx, client in clients.iterrows():
        cid = int(idx)
        all_records.extend(
            build_record(
                scorer,
                cid,
                client,
                top_k=args.top_k,
                explain_k=args.explain_k,
                render_prompt=args.prompts,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    flat = []
    for r in all_records:
        row = {k: v for k, v in r.items() if k != "sales_prompt"}
        row["top_features_json"] = json.dumps(r["top_features"], ensure_ascii=False)
        flat.append(row)
    pd.DataFrame(flat).to_csv(args.csv, index=False)

    print(f"Clients scored: {len(clients)}")
    print(f"Records: {len(all_records)} -> {args.output}")
    print(f"CSV -> {args.csv}")


if __name__ == "__main__":
    main()
