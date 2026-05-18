# Архитектура модели оценки склонности к продуктам (онбординг)

## Контекст в цепочке задач

```
┌─────────────────────────┐     ┌──────────────────────────────┐     ┌─────────────────────────┐
│ Классификатор P1–P8     │     │ Модели склонности (продукт)  │     │ LLM: sales-аргументы    │
│ приоритизация сценариев │ ──► │ score / топ-признаки         │ ──► │ под сегмент + клиента   │
│ alfa_onboarding_*       │     │ propensity_synthetic + LGBM  │     │ BRD + Jinja-промпты     │
└─────────────────────────┘     └──────────────────────────────┘     └─────────────────────────┘
```

**Связь двух обучений (практический вывод):** полная сквозная согласованность не обязательна, но **популяция клиентов и сегмент P1–P8 должны быть общими**. Синтетика склонности строится поверх тех же 5000 клиентов; `priority_segment` входит в признаки и в правила генерации меток — так аргументы «почему ЗПП» не противоречат тому, что клиенту уже назначен приоритетный сценарий.

Для новых юрлиц точность склонности низка первые ~30 дней (см. summary) — в синтетике это отражено штрафом по `client_days_proxy` (`days_from_ogrn`).

## Продукты со склонностью

| `product_id`        | Сценарий онбординга              | Якорный | Источник топ-признаков (BRD)     |
|---------------------|----------------------------------|---------|----------------------------------|
| `zpp`               | Зарплатный проект                | нет     | segment, okved, ogrn, payment FL |
| `alfa_payments`     | Подписка Альфа-платежи           | да      | sum_deb_fl, balance, okved       |
| `nachalo`           | Подписка «Начало»                | да      | commission, cred/deb, balance    |
| `trade_acquiring`   | Торговый эквайринг               | нет     | retail OKVED, turnover           |
| `internet_acquiring`| Интернет-эквайринг              | нет     | digital categ, online source     |
| `tax_jar`           | Налоговая копилка                | да      | ИП/УСН proxy, tax categ          |
| `savings`           | Накопительный счёт               | да      | balance, stability               |
| `accounting`        | Бухгалтерия                      | да      | ИП, young UL, low complexity     |

«Действия» (мобильное приложение, платёж по карте) **без отдельной модели склонности** — приоритет задаётся правилами онбординга (классификатор P*).

## Формат данных

**Long-формат:** одна строка = клиент × продукт.

| Поле | Описание |
|------|----------|
| `client_id` | Индекс строки в `alfa_onboarding_dataset_5000.csv` |
| `priority_segment` | P1–P8 из первого классификатора |
| `product_id` | Код продукта |
| `propensity_score` | Непрерывная склонность [0, 1] |
| `propensity_label` | Бинарная метка (score ≥ 0.5) |
| `label_source` | `hybrid_rule+mock_llm` (или `rule+noise` при `--mode rule`) |
| … | Общие признаки клиента + `product_id` (категория) |

Объём: 5000 × 8 продуктов = **40 000** строк (файл `data/propensity_synthetic.csv`).

## Модель

- **Алгоритм:** LightGBM binary classifier (одна модель на все продукты, `product_id` — категориальный признак).
- **Цель:** воспроизвести синтетические метки и дать интерпретируемые важности признаков для сегментации под LLM (BRD, п. 6–7).
- **Артефакты:** `models/propensity_lgbm.pkl`, `models/feature_config.json`.
- **Валидация:** hold-out 20%, метрики ROC-AUC / PR-AUC, важности по `product_id`.

Альтернатива (не в пилоте): отдельная модель на продукт — выше точность, сложнее сопровождение.

## Генерация синтетики (режим `hybrid`, по умолчанию)

1. Загрузка `alfa_onboarding_dataset_5000.csv` — **те же client_id и P1–P8**, что классификатор ступени 1.
2. **Правила** (`feature_rules.py`): BRD-логика + `SEGMENT_PRODUCT_BIAS` + шум → базовый score.
3. **Mock-LLM** (`propensity_llm_mock.py`): Jinja-промпт с задачей 1, примерами сегментов, BRD-фичами → уточнение score/label и `top_features_json`.
4. **Согласованность** (`segment_consistency.py`): boost якорных продуктов портрета; метрика в `data/consistency_report.json`.
5. `label_source` = `hybrid_rule+mock_llm`; примеры промптов — `data/llm_prompts_sample.jsonl`.

```bash
python3 src/generate_propensity_synthetic.py --mode hybrid
```

## Использование для sales-аргументов

1. Скоринг клиента → топ-N продуктов по `propensity_score`.
2. SHAP / `feature_importances_` → 3–5 топ-признаков на пару (клиент, продукт).
3. В промпт LLM: продукт, score, сегмент P*, признаки, шаблон из BRD (формат 3–4 предложения).

## Пайплайн скоринга и sales-аргументов (этап 2)

```
clients (5000)
    → PropensityScorer.score_client(top_k=3)
    → explain_client_product (SHAP, top-5 признаков)
    → render_sales_prompt (Jinja: sales_argument_final.j2)
    → generate_sales_argument (mock | OPENAI_API_KEY)
    → data/sales_arguments.jsonl
```

**Скрипты:**

| Команда | Назначение |
|---------|------------|
| `python3 src/batch_score_clients.py --limit 0 --prompts` | Все клиенты, топ-3, SHAP, промпты |
| `python3 src/generate_sales_arguments.py --limit 0` | Mock-аргументы |
| `python3 src/generate_sales_arguments.py --llm` | Через OpenAI-compatible API |
| `python3 src/train_per_product.py` | 8 моделей в `models/per_product/` |
| `bash run_pipeline.sh` | Полный прогон |

**Выходы:** `data/scored_clients_top3.jsonl`, `data/scored_clients_top3.csv`, `data/sales_arguments.jsonl`

## Per-product модели

Дополнительно к единой модели обучены отдельные LGBM без `product_id` в признаках — метрики в `models/per_product/metrics.json`. Использовать, если нужна максимальная точность по одному продукту в проде.

## LLM и BRD-скрипты

Источник: `BRD сэйлз-аргументы.pdf` → структурировано в:
- `config/brd_scripts.yaml` — скрипты предпродажи (ЗПП, ТЭ, ИЭ, подписки, налоговая копилка…), сегменты ЗПП, комплаенс
- `config/onboarding_features.yaml` — `selected_features` для классификатора P1–P8

Промпт `sales_argument_final.j2` подставляет: вопросы на выявление потребности, заход, преимущества, пример варианта 2 для подписок.

- **Mock:** текст в стиле BRD (`generator: mock+brd`)
- **API:** `OPENAI_API_KEY` + флаг `--llm`
- Пример рендера: `python3 prompts/render_brd_example.py`

## Структура репозитория (папка `Николай`)

```
Николай/
├── ARCHITECTURE.md
├── run_pipeline.sh
├── config/products.yaml
├── prompts/
│   ├── system_context.j2
│   ├── generate_client_product_row.j2
│   └── sales_argument_final.j2
├── src/
│   ├── feature_rules.py
│   ├── generate_propensity_synthetic.py
│   ├── train_propensity.py
│   ├── train_per_product.py
│   ├── scoring.py
│   ├── explain.py
│   ├── batch_score_clients.py
│   ├── llm_sales_pipeline.py
│   └── generate_sales_arguments.py
├── data/
│   ├── alfa_onboarding_dataset_5000.csv
│   ├── propensity_synthetic.csv
│   ├── consistency_report.json
│   └── llm_prompts_sample.jsonl
├── summary_onboarding_sales.txt
├── models/
│   ├── propensity_lgbm.pkl
│   ├── feature_config.json
│   └── per_product/*.pkl
└── notebooks/
    ├── train_propensity_model.ipynb
    └── scoring_and_sales_pipeline.ipynb
```
