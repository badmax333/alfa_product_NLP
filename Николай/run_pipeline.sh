#!/usr/bin/env bash
# Полный пайплайн: синтетика → обучение → скоринг → аргументы
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. Synthetic propensity (hybrid: rules + mock-LLM + P* consistency) =="
python3 src/generate_propensity_synthetic.py --mode hybrid

echo "== 2. Train unified model =="
python3 src/train_propensity.py

echo "== 3. Train per-product models =="
python3 src/train_per_product.py

echo "== 4. Batch score (all clients, top-3, SHAP, prompts) =="
python3 src/batch_score_clients.py --limit 0 --prompts

echo "== 5. Sales arguments (mock; use --llm in generate_sales_arguments for API) =="
python3 src/generate_sales_arguments.py --limit 0

echo "Done. See data/ and models/"
