# Ступень 2 — Скоринг склонности к продуктам

После первого взаимодействия (Ступень 1) накапливаются данные о реакции клиента.
На их основе рассчитывается склонность к 8 продуктам — какой предложить следующим.

## Состояние данных

| Параметр | Статус |
|---|---|
| Регистрационные данные | ✅ Те же 8 признаков, что на Ступени 1 |
| Результат взаимодействия | ✅ `interest_score` из метрик Ступени 1 |
| LightGBM-модель | ✅ `models/propensity_lgbm.pkl` (ROC-AUC 0.984, PR-AUC 0.988) |
| Транзакционная история | 🔲 Пока не подключена — fallback на rule-based логику |

## Pipeline

```
Классификация Ступени 1 (портрет P1–P8)
        +
Метрики взаимодействия (interest_score)
        +
Признаки клиента
        │
        ▼
score_propensity()   ← services/propensity_scorer.py
        │
        ├─► LightGBM pipeline (если models/propensity_lgbm.pkl есть)
        │         └─ калибровка logit + поправка на interest_score
        │
        └─► Rule-based fallback (8 rule-функций, по одной на продукт)
                  └─ сигмоид(base_logit + segment_bias + maturity_penalty + interaction_delta)
        │
        ▼
Top-K продуктов с propensity_score + top_factors (до 5 факторов с direction + reason)
```

## 8 продуктов

Определены в `config/propensity.py` → `PROPENSITY_PRODUCTS`:

| ID | Название | AME |
|---|---|---|
| `zpp` | Зарплатный проект | 10 |
| `alfa_payments` | Подписка Альфа-платежи | 6 |
| `nachalo` | Подписка «Начало» | 6 |
| `trade_acquiring` | Торговый эквайринг | 7 |
| `internet_acquiring` | Интернет-эквайринг | 11 |
| `tax_jar` | Налоговая копилка | 14 |
| `savings` | Накопительный счет | — |
| `accounting` | Бухгалтерия | 12 |

## Rule-based скоринг

Для каждого продукта — своя функция `_score_<product>()`, которая:
1. Начинает с `base_logit` (от −1.2 до −0.5)
2. Добавляет/убирает очки по признакам (`smb_type_code`, `okved_major`, `categ_name`, обороты, уже подключённые продукты)
3. Применяет штраф за молодой бизнес (`days_from_ogrn < 30/90`)
4. Добавляет `SEGMENT_PRODUCT_BIAS` — поправку на портрет P1–P8
5. Добавляет `interaction_delta` = `(interest_score − 0.5) × 0.35` из Ступени 1
6. Переводит итоговый logit в вероятность через sigmoid

## LightGBM-модель

**Артефакт:** `models/propensity_lgbm.pkl`  
**Конфиг признаков:** `models/feature_config.json`  
**ROC-AUC:** 0.984 · **PR-AUC:** 0.988

Признаки: 10 категориальных (`product_id`, `priority_segment`, `smb_type_code`, `okved_major_wrapped`, `categ_name`, `srvpackage_sale_uk`, `sourceattr_ccode`, `city`, `addrf_region_name`, `division_name`) + 26 числовых.

Если файл модели отсутствует — автоматически используется rule-based fallback. Поле `model_source` в ответе указывает, какой метод был применён.

## API

```http
POST /api/v1/propensity/score
{
  "classification": {"predicted_class": "P1", ...},
  "client_features": {"smb_type_code": "2", ...},
  "metrics_result": {"interest_score": 0.72, ...},
  "top_k": 3
}
→ {
    "portrait": "P1",
    "portrait_label": "Розничный продавец",
    "model_source": "lightgbm_propensity_lgbm",
    "interaction_interest_score": 0.72,
    "top_products": [
      {
        "product_id": "zpp",
        "product_name": "Зарплатный проект",
        "propensity_score": 0.81,
        "rank": 1,
        "top_factors": [
          {"feature": "smb_type_code", "impact": 0.6, "direction": "increases", "reason": "..."},
          ...
        ]
      },
      ...
    ],
    "all_products": [...]
  }
```

## Что передаётся в LLM на Ступени 2 (следующий шаг)

После получения `top_products` генерируется sales-аргумент уже под конкретный продукт.
В промпт планируется передавать:

```
Портрет клиента (P1–P8) + описание
Признаки клиента (регистрационные)
Sales-аргумент Ступени 1 + как отреагировал (interest_score, метрики)
Предлагаемый продукт (название, AME-код)
Top-3 фактора склонности (feature + reason) — «почему именно этот продукт»
Тип взаимодействия (banner / push / voice)
```

## Roadmap

- ✅ Скоринг склонности к 8 продуктам (LightGBM + rule-based fallback)
- ✅ Top-K факторов с direction + reason
- ✅ Учёт interest_score из Ступени 1
- 🔲 Генерация sales-аргумента Ступени 2 (отдельный промпт под каждый продукт)
- 🔲 LLM Compliance-checker
- 🔲 Батч-генерация аргументов (раз в сутки) → CRM
- 🔲 Подключение реальных транзакционных признаков
- 🔲 A/B тест: персонализированный vs стандартный скрипт
