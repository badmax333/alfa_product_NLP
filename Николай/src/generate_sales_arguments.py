#!/usr/bin/env python3
"""Генерация sales-аргументов по scored_clients_top3.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from explain import FeatureContribution
from llm_sales_pipeline import generate_sales_argument
from scoring import load_onboarding_clients

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "scored_clients_top3.jsonl"
DEFAULT_OUT = ROOT / "data" / "sales_arguments.jsonl"


def _feats_from_record(raw: list[dict]) -> list[FeatureContribution]:
    return [
        FeatureContribution(
            feature=x["feature"],
            label_ru=x["label_ru"],
            value=x["value"],
            shap_value=x["shap_value"],
            direction=x["direction"],
        )
        for x in raw
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--llm", action="store_true", help="Use OpenAI API (needs OPENAI_API_KEY)")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    use_mock = not args.llm
    clients = load_onboarding_clients()
    records_in = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            records_in.append(json.loads(line))
    if args.limit > 0:
        seen = set()
        filtered = []
        for r in records_in:
            if r["client_id"] not in seen:
                seen.add(r["client_id"])
                if len(seen) > args.limit:
                    break
            if len(seen) <= args.limit:
                filtered.append(r)
        records_in = filtered

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for rec in records_in:
            cid = rec["client_id"]
            client = clients.iloc[cid]
            feats = _feats_from_record(rec["top_features"])
            arg = generate_sales_argument(
                client_id=cid,
                client=client,
                product_id=rec["product_id"],
                product_name=rec["product_name"],
                propensity_score=rec["propensity_score"],
                top_features=feats,
                use_mock=use_mock,
            )
            merged = {**rec, "sales_content": arg}
            out.write(json.dumps(merged, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records_in)} arguments -> {args.output} (mock={use_mock})")


if __name__ == "__main__":
    main()
