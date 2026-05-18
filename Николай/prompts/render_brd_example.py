#!/usr/bin/env python3
"""Пример промпта с BRD-скриптом для client_id=0, product=zpp."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from explain import explain_client_product
from llm_sales_pipeline import render_sales_prompt
from scoring import PropensityScorer, load_onboarding_clients

clients = load_onboarding_clients()
client = clients.iloc[0]
scorer = PropensityScorer()
top = scorer.score_client(client, top_k=1)[0]
feats = explain_client_product(
    scorer.pipe, client, top.product_id,
    cat_features=scorer.cat_features, num_features=scorer.num_features,
)
prompt = render_sales_prompt(
    client_id=0, client=client, product_id=top.product_id,
    product_name=top.product_name, propensity_score=top.propensity_score,
    top_features=feats,
)
out = ROOT / "prompts" / "_example_rendered_brd_zpp.txt"
out.write_text(prompt, encoding="utf-8")
print(f"Written -> {out}")
