# Модель склонности к продуктам (онбординг) — ступень 2

## Статус задач

| Идея | Статус | Где |
|------|--------|-----|
| Синтетика на тех же клиентах, что P1–P8 | `data/propensity_synthetic.csv`, `generate_propensity_synthetic.py` |
| Связь двух обучений (правдоподобность) | `segment_profiles.yaml`, `segment_consistency.py`, `priority_segment` в модели, `data/consistency_report.json` |
| Метки склонности через mock-LLM + Jinja + BRD  | `propensity_llm_mock.py`, `prompts/`, `label_source=hybrid_rule+mock_llm` |
| Контекст LLM: задача 1 + синт онбординга + BRD  | `system_context.j2` + примеры сегментов в промпте, `data/llm_prompts_sample.jsonl` |

## Готовые артефакты

| № | Артефакт | Путь |
|---|----------|------|
| 0 | Архитектура | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 1 | Синтетический датасет | [data/propensity_synthetic.csv](data/propensity_synthetic.csv) |
| 2 | Ноутбук обучения | [notebooks/train_propensity_model.ipynb](notebooks/train_propensity_model.ipynb) |
| 3 | Веса модели | [models/propensity_lgbm.pkl](models/propensity_lgbm.pkl) |

## Быстрый старт

```bash
cd Николай
pip install -r requirements.txt

# 1. Синтетика (hybrid: правила + mock-LLM + согласованность P*)
python3 src/generate_propensity_synthetic.py --mode hybrid

# 2. Обучение
python3 src/train_propensity.py
# или notebooks/train_propensity_model.ipynb

# 3. Переобучить после новой синтетики — обязательно шаг 2
```

Полный пайплайн (включая скоринг и аргументы): `bash run_pipeline.sh`

## Режимы синтетики

- `--mode hybrid` (по умолчанию) — правила BRD → mock-LLM по Jinja → boost якорных продуктов P*
- `--mode rule` — только `feature_rules.py` (без LLM-слоя)

## Входные данные

- `data/alfa_onboarding_dataset_5000.csv` — клиенты и сегменты P1–P8 (классификатор ступени 1)
- BRD: `config/brd_scripts.yaml`, `config/products.yaml`, `config/onboarding_features.yaml`

## Интеграция (ступень 2 → общий пайплайн)

Минимум для скоринга без переобучения:

| Файл | Назначение |
|------|------------|
| `models/propensity_lgbm.pkl` | веса LightGBM |
| `models/feature_config.json` | список `cat_features` / `num_features` |
| `config/products.yaml` | 10 продуктов и названия |

Бинарные колонки (`abm_entered`, `mobile_app_entered`, `plastic_card_issued`, `cashback_selected`) **добавляются автоматически** в `PropensityScorer` / `load_onboarding_clients()`, если их нет во входном CSV.

```python
from scoring import PropensityScorer, load_onboarding_clients
scorer = PropensityScorer()
client = load_onboarding_clients().iloc[0]
print(scorer.score_client(client, top_k=3))
```

Стек: LightGBM + sklearn pipeline (см. `requirements.txt`). Демо в контейнере: `docker build -t propensity .`

## Продукты (10)

zpp, alfa_payments, nachalo, trade_acquiring, internet_acquiring, tax_jar, savings, accounting, **business_card**, **mobile_app**
