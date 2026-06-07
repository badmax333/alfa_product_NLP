# Alfa Bank — Smart Onboarding для бизнес-клиентов

POC персонализированного онбординга юридических лиц и ИП на базе ML + LLM.

![demo](docs/demo.gif)

## Обзор проекта

**Проблема:** Бизнес-клиенты (ЮЛ, ИП, КФХ) открывают расчётный счёт в банке, но не активируют банковские продукты — остаются в статусе «спящих». Традиционный онбординг использует единый скрипт для всех типов клиентов, игнорируя отраслевую специфику, размер бизнеса и реальные потребности.

**Решение:** Трёхступенчатая система персонализированного онбординга на базе ML + LLM. На каждом этапе жизненного цикла клиента система применяет релевантную модель данных и генерирует персонализированный аргумент для активации нужного продукта.

**Цель POC:** Подтвердить гипотезу о том, что персонализированный аргумент, сгенерированный на основе профиля клиента, повышает конверсию в активацию якорного продукта по сравнению с базовым скриптом.

```
Клиент открыл счёт
        │
        ▼
┌───────────────────┐    День 0–4     ┌─────────────────────────────┐
│   СТУПЕНЬ 1       │ ──────────────► │  Статический классификатор  │
│  Холодный старт   │                 │  (CatBoost, 8 сегментов)    │
│  Регистр. данные  │                 │  -> Якорный продукт + скрипт │
└───────────────────┘                 └─────────────────────────────┘
        │
        ▼ (накопление транзакций 4–30 дней)
┌───────────────────┐    День 4–30    ┌─────────────────────────────┐
│   СТУПЕНЬ 2       │ ──────────────► │  Модели склонности (AUC>0.65)│
│  Транзакц. история│                 │  SHAP -> LLM-аргумент        │
│  30+ событий      │                 │  (батч-генерация, 1/сутки)  │
└───────────────────┘                 └─────────────────────────────┘
        │
        ▼ (полная история + LTV)
┌───────────────────┐    День 30+     ┌─────────────────────────────┐
│   СТУПЕНЬ 3       │ ──────────────► │  Сегментация по статусу     │
│  Полный профиль   │                 │  Удержание / добор / апсейл │
│  LTV + активность │                 │  Расчёт упущенной выгоды    │
└───────────────────┘                 └─────────────────────────────┘
```

---

## Документация

Детальная документация по каждой ступени в папке `docs/`:

| Раздел | Файл | Что внутри |
|---|---|---|
| Ступень 1: Обзор | [docs/stage1/overview.md](docs/stage1/overview.md) | Проблема холодного старта, pipeline |
| Ступень 1: Классификатор | [docs/stage1/classifier.md](docs/stage1/classifier.md) | CatBoost, 50 признаков, 8 портретов, SHAP |
| Ступень 1: Sales-аргументы | [docs/stage1/sales_arguments.md](docs/stage1/sales_arguments.md) | Типы взаимодействия, промпты, где что лежит |
| Ступень 1: Метрики | [docs/stage1/metrics.md](docs/stage1/metrics.md) | 35 метрик по 5 уровням, генерация LLM/random |
| Ступень 2: Скоринг склонности | [docs/stage2/overview.md](docs/stage2/overview.md) | 8 продуктов, LightGBM + rule-based, API |


---

## Быстрый старт

```bash
# 1. Создать и активировать venv
python -m venv .venv && source .venv/bin/activate

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Настроить Mistral API (нужен только для LLM-режима в Tab 3)
cp .env.example .env
# вставьте MISTRAL_API_KEY=... в .env

# 4. Запустить сервер
bash run.sh
```

`run.sh` выставляет `PYTHONPATH` и запускает `uvicorn api.app:app --reload --host 0.0.0.0 --port 8000`.  
Альтернатива: `python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000` (из корня проекта).

Открыть: [http://localhost:8000](http://localhost:8000)

---

## Веб-интерфейс

| Вкладка | Что делает |
|---|---|
| **1 — Stage 1: Классификация** | Выбрать/настроить профиль -> получить портрет (P1–P8), уверенность модели, SHAP-значения, якорный продукт |
| **2 — Stage 1: Sales-аргумент** | Выбрать тип взаимодействия (banner/push/voice) -> увидеть промпт -> сгенерировать аргумент через Mistral |
| **3 — Stage 1: Метрики** | Канал определяется автоматически -> сгенерировать 20 (digital) или 15 (voice) метрик взаимодействия |
| **4 — Stage 2: Склонность** | Рассчитать top-3 продуктов по LightGBM с учётом interest_score из шага 3 |
| **5 — Stage 2: Sales-аргумент** | Выбрать продукт из top-3 и тип взаимодействия -> сгенерировать аргумент с учётом истории шагов 1–3 |
| **6 — Stage 2: Метрики** | Оценить реакцию клиента на аргумент Stage 2 (Mistral LLM или случайная симуляция) |

---

## API

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/api/v1/config` | Конфигурация UI (признаки, пресеты, описания классов) |
| POST | `/api/v1/predict` | Классификация клиента (CatBoost + SHAP) |
| GET | `/api/v1/sales-args/config` | Типы взаимодействия + демо-примеры аргументов |
| POST | `/api/v1/sales-args/render-prompt` | Рендеринг промпта Stage 1 аргумента (Jinja2, без вызова LLM) |
| POST | `/api/v1/sales-args/generate` | Генерация персонализированного Stage 1 аргумента через Mistral |
| POST | `/api/v1/metrics/render-prompt` | Рендеринг промпта метрик без вызова Mistral |
| POST | `/api/v1/metrics/generate` | Генерация метрик (`method: "llm"` или `"random"`) — используется для Ступеней 1 и 2 |
| POST | `/api/v1/propensity/score` | Скоринг склонности клиента к 8 продуктам после метрик Ступени 1 |
| POST | `/api/v1/sales-args/render-prompt-stage2` | Рендеринг промпта Stage 2 аргумента (Jinja2, без вызова LLM) |
| POST | `/api/v1/sales-args/generate-stage2` | Генерация Stage 2 аргумента с учётом истории взаимодействия и скоринга склонности |

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Батч-генерация и оценка качества

### Батч-генерация Stage 1

```bash
python -m pipeline.stage1_pipeline
```

```python
from pipeline.stage1_pipeline import run_batch
results = run_batch(n=100, method="random")
```

### Полный двухступенчатый пайплайн

```python
from pipeline.full_pipeline import full_run_single

# Один клиент — быстрый режим (без LLM для метрик)
result = full_run_single(metrics_method="random")

# Полный LLM-режим
result = full_run_single(s1_personalized=True, s2_personalized=True, metrics_method="llm")

# Структура результата
result["summary"]  # portrait, s1_interest, s2_interest, s1_activated, s2_activated
result["s1_argument"]["headline"]  # текст аргумента Ступени 1
result["propensity"]["top_products"]  # top-3 продукта по склонности
result["s2_argument"]["headline"]  # текст аргумента Ступени 2
```

### Оценка качества аргументов (персонализированные vs обезличенные)

```bash
# n=20 клиентов
python -m pipeline.evaluation

# Быстрый тест
python -m pipeline.evaluation --n 5

# Сохранить результаты
python -m pipeline.evaluation --n 50 --output results/eval_50.json
```

Сравнивает две стратегии на одних клиентах, метрики в обоих случаях через LLM:
- **Персонализированные:** LLM создаёт аргумент под портрет -> Mistral оценивает реакцию
- **Обезличенные:** фиксированный шаблон аргумента -> Mistral оценивает реакцию

Если delta `interest_score` > 0 — персонализация измеримо улучшает качество аргументов.

---

## Структура репозитория

```
alfa_product_NLP/
├── api/
│   ├── app.py          # FastAPI: 10 эндпоинтов (классификация, аргументы, метрики, склонность)
│   └── schemas.py      # Pydantic-схемы запросов и ответов
├── config/             # Константы: признаки, метрики, аргументы, продукты склонности
├── docs/               # Подробная документация
│   ├── stage1/         # Классификатор, аргументы, метрики
│   └── stage2/         # Скоринг склонности, Stage 2 аргументы
├── models/
│   ├── classifier.py       # CatBoost-классификатор (8 портретов + SHAP)
│   ├── propensity_lgbm.pkl # LightGBM-модель склонности (ROC-AUC 0.984)
│   └── feature_config.json # Конфиг признаков LightGBM
├── pipeline/
│   ├── stage1_pipeline.py  # Батч-генерация данных Ступени 1
│   ├── full_pipeline.py    # Полный двухступенчатый пайплайн (full_run_single)
│   └── evaluation.py       # Оценка качества: персонализированные vs обезличенные аргументы
├── prompts/            # Jinja2-шаблоны для LLM
│   ├── stage1_sales_argument.j2
│   ├── stage1_metrics_generation.j2
│   └── stage2_sales_argument.j2
├── services/
│   ├── llm.py                      # Mistral клиент + retry при 429
│   ├── sales_argument_generator.py # Генерация аргументов Stage 1 и Stage 2
│   ├── sales_arg_renderer.py       # Рендеринг Jinja2-промптов аргументов
│   ├── metrics_generator.py        # Генерация метрик через Mistral
│   ├── random_metrics_generator.py # Случайная симуляция метрик (без LLM)
│   └── propensity_scorer.py        # Скоринг склонности (LightGBM + rule-based)
└── web/                # 6-вкладочный SPA (HTML + JS + CSS)
```

---

## Roadmap

**Ступень 1 (реализовано)**
- CatBoost классификатор (8 портретов) + SHAP
- Генерация sales-аргументов через Mistral (banner / push / voice)
- FastAPI (10 эндпоинтов) + 6-вкладочный SPA
- 5-уровневая система метрик (20 digital + 15 voice)
- Генерация метрик: Mistral LLM и случайная (без API)
- Батч-генерация синтетических данных (pipeline/stage1_pipeline.py)
- Retry при ошибках 429 (services/llm.py)

**Ступень 2 (реализовано)**
- Скоринг склонности к 8 продуктам (LightGBM ROC-AUC 0.984 + rule-based fallback)
- Учёт `interest_score` из метрик взаимодействия Ступени 1
- Top-K факторов с direction + reason
- Генерация Stage 2 аргумента с контекстом истории (промпт `stage2_sales_argument.j2`)
- Генерация метрик Stage 2 (те же 20/15 метрик, тот же эндпоинт)
- Полный двухступенчатый пайплайн (pipeline/full_pipeline.py)
- Оценка качества аргументов (pipeline/evaluation.py)