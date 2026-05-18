# Модель склонности к продуктам (онбординг) — сдача

## Статус задач

| Идея | Статус | Где |
|------|--------|-----|
| Синтетика на тех же клиентах, что P1–P8 | `data/propensity_synthetic.csv`, `generate_propensity_synthetic.py` |
| Связь двух обучений (правдоподобность) | `segment_profiles.yaml`, `segment_consistency.py`, `priority_segment` в модели, `data/consistency_report.json` |
| Метки склонности через mock-LLM + Jinja + BRD  | `propensity_llm_mock.py`, `prompts/`, `label_source=hybrid_rule+mock_llm` |
| Контекст LLM: задача 1 + синт онбординга + BRD  | `system_context.j2` + примеры сегментов в промпте, `data/llm_prompts_sample.jsonl` |

## Обязательные артефакты (ТЗ)

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
