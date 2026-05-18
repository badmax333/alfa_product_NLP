#!/usr/bin/env python3
"""Пример рендера Jinja-промпта (без вызова LLM)."""

from pathlib import Path

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
env = Environment(loader=FileSystemLoader(ROOT / "prompts"), autoescape=False)

clients = pd.read_csv(ROOT / "data" / "alfa_onboarding_dataset_5000.csv").head(1)
row = clients.iloc[0]

with open(ROOT / "config" / "products.yaml") as f:
    cfg = yaml.safe_load(f)

product_id = "zpp"
product_cfg = cfg["products"][product_id]

template = env.get_template("generate_client_product_row.j2")
prompt = template.render(
    products=list(cfg["products"].keys()),
    client_id=0,
    priority_segment=row["target"],
    smb_type_code=int(row["smb_type_code"]),
    okved_major=int(row["okved_major"]),
    categ_name=row["categ_name"],
    city=row["city"],
    days_from_ogrn=int(row["days_from_ogrn"]),
    week_sum_transactions=float(row["week_sum_transactions"]),
    srvpackage_sale_uk=row["srvpackage_sale_uk"],
    product_id=product_id,
    product_name=product_cfg["name_ru"],
    brd_features=product_cfg["brd_features"],
)

out = ROOT / "prompts" / "_example_rendered_zpp.txt"
out.write_text(prompt, encoding="utf-8")
print(f"Written example prompt -> {out}")
