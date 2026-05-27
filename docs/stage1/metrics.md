# Метрики оценки взаимодействия (Ступень 1)

## 5-уровневая воронка

Метрики организованы по 5 уровням воронки — от доставки до негативной реакции:

| Уровень | Название | Смысл |
|---|---|---|
| 1 | Доставка / контакт | Аргумент был доставлен и замечен клиентом |
| 2 | Внимание / интерес | Клиент проявил интерес (клик, вопрос, пролистывание) |
| 3 | Целевое действие | Клиент начал оформление (заявка, устное согласие) |
| 4 | Активация / использование | Продукт подключён и использован |
| 5 | Качество / негатив | Отказ, жалоба, отписка, раздражение |

Определения уровней и имена — в `config/metrics.py` → `LEVEL_NAMES`.

---

## Цифровой канал — 20 метрик

Определены в `config/metrics.py` → `DIGITAL_METRICS`.

| Метрика | Уровень | Тип | Диапазон | Что измеряет |
|---|---|---|---|---|
| `banner_shown` | 1 | binary | 0/1 | Баннер/сообщение показан в интерфейсе |
| `banner_visible_sec` | 1 | duration_sec | 0–120 | Секунд баннер в зоне видимости |
| `push_delivered` | 1 | binary | 0/1 | Push/Email доставлен |
| `push_opened` | 1 | binary | 0/1 | Push/Email открыт |
| `banner_clicked` | 2 | binary | 0/1 | Клик по баннеру или CTA |
| `product_page_visited` | 2 | binary | 0/1 | Переход на страницу продукта |
| `time_on_product_page_sec` | 2 | duration_sec | 0–600 | Секунд на странице продукта |
| `scroll_depth_pct` | 2 | float | 0–100 | Глубина прокрутки карточки (%) |
| `remind_later_clicked` | 2 | binary | 0/1 | «Напомнить позже» — интерес без готовности |
| `repeated_view_count` | 2 | integer | 0–10 | Повторных просмотров карточки |
| `application_started` | 3 | binary | 0/1 | Начало заполнения заявки |
| `application_completed` | 3 | binary | 0/1 | Заявка отправлена |
| `time_to_action_min` | 3 | float | 0–10080 | Минут от показа до действия |
| `product_activated` | 4 | binary | 0/1 | Продукт фактически подключён |
| `first_transaction_done` | 4 | binary | 0/1 | Первая транзакция через продукт |
| `days_to_first_use` | 4 | integer | 0–90 | Дней от активации до первого использования |
| `banner_dismissed` | 5 | binary | 0/1 | Баннер намеренно закрыт |
| `push_unsubscribed` | 5 | binary | 0/1 | Отписка от push/email |
| `complaint_filed` | 5 | binary | 0/1 | Жалоба на нерелевантность |
| `negative_rating` | 5 | integer | 1–5 | Оценка предложения (1 = плохо, 5 = отлично) |

---

## Голосовой канал — 15 метрик

Определены в `config/metrics.py` → `VOICE_METRICS`.

| Метрика | Уровень | Тип | Что измеряет |
|---|---|---|---|
| `call_connected` | 1 | binary | Дозвон состоялся |
| `call_duration_sec` | 1 | duration_sec | Длительность разговора (сек) |
| `reached_argument_block` | 1 | binary | Оператор дошёл до блока аргумента |
| `client_didnt_hangup_before_argument` | 1 | binary | Клиент не повесил трубку до аргумента |
| `positive_reaction` | 2 | binary | Позитивная / нейтрально-заинтересованная реакция |
| `clarifying_questions_count` | 2 | integer | Уточняющих вопросов задано (прокси интереса) |
| `requested_link_or_materials` | 2 | binary | Просьба прислать ссылку / материалы |
| `agreed_to_callback` | 2 | binary | Согласие на перезвон |
| `verbal_agreement` | 3 | binary | Устное согласие на подключение |
| `followed_link_after_call` | 3 | binary | Переход по ссылке после звонка |
| `product_connected_after_call` | 4 | binary | Продукт подключён в течение 7 дней после звонка |
| `first_operation_after_call` | 4 | binary | Первая операция через подключённый продукт |
| `negative_reaction_voice` | 5 | binary | Негативная реакция / раздражение |
| `complaint_after_call` | 5 | binary | Жалоба после звонка |
| `requested_no_more_calls` | 5 | binary | Просьба больше не звонить |

---

## Поведенческие профили портретов

В `config/metrics.py` → `PORTRAIT_BEHAVIORAL_PROFILES` для каждого из 8 портретов задан:

| Поле | Тип | Описание |
|---|---|---|
| `digital_affinity` | enum | very_low / low / medium / high / very_high |
| `voice_affinity` | enum | very_low / low / medium / high / very_high |
| `interest_base` | float 0–1 | Базовый уровень интереса (стартовая точка генерации) |
| `typical_behavior` | str | Описание поведения — передаётся в промпт Mistral |
| `negative_triggers` | str | Что вызывает негатив — также идёт в промпт |

Краткая сводка:

| Портрет | digital | voice | interest_base |
|---|---|---|---|
| P1 Розничный | medium | high | 0.72 |
| P2 IT | very_high | low | 0.65 |
| P3 Строитель | low | very_high | 0.68 |
| P4 Фрилансер | high | low | 0.55 |
| P5 Стартап | high | medium | 0.60 |
| P6 HoReCa | medium | medium | 0.70 |
| P7 Логистика | medium | high | 0.66 |
| P8 Агро | low | very_high | 0.62 |

---

## Генерация метрик через LLM (Mistral)

**Файл:** `services/metrics_generator.py`  
**Промпт:** `prompts/stage1_metrics_generation.j2`  
**Модель:** `mistral-large-latest`  
**Ключ:** `MISTRAL_API_KEY` из `.env`

### Pipeline

```
client_features + classification + sales_argument + channel
        │
        ▼
render_metrics_prompt()    ← Jinja2: prompts/stage1_metrics_generation.j2
        │
        ▼
Mistral API
        │
        ▼
_extract_json()            ← парсинг JSON из ответа
        │
        ▼
_apply_noise()             ← ±15% шум к числовым метрикам
        │
        ▼
structured metrics list    ← список MetricValueItem
```

### Что передаётся в промпт

- Портрет клиента + поведенческий профиль (`typical_behavior`, `negative_triggers`)
- Top-5 SHAP-факторов
- Sales-аргумент (headline + body)
- Канал (digital / voice)
- Полное описание всех метрик с диапазонами

### Что возвращает Mistral

```json
{
  "interest_score": 0.72,
  "user_reaction_text": "Клиент кликнул по баннеру...",
  "metrics": {
    "banner_shown": 1,
    "banner_clicked": 1,
    "product_page_visited": 0,
    ...
  }
}
```

---

## Случайная генерация (без LLM)

**Файл:** `services/random_metrics_generator.py`

Генерация без API-вызова на основе поведенческих профилей. Используется для быстрого тестирования и батч-генерации.

### Алгоритм

1. `interest_score` = `interest_base` + гауссов шум (σ=0.08), clamp в [0, 1]
2. `affinity` переводится в числовой коэффициент: very_low=0.2, low=0.35, medium=0.55, high=0.75, very_high=0.90
3. Метрики генерируются каскадно с логической зависимостью

### Каскадная логика (digital)

```
banner_shown=1 (p = interest × affinity)
  └─ banner_visible_sec: uniform(2, 60) если shown=1, иначе 0
  └─ push_delivered=1 (p = 0.95)
       └─ push_opened=1 (p = interest × 0.6)
  └─ banner_clicked=1 (p = interest × affinity × 0.5)
       └─ product_page_visited=1 (p = interest × 0.7)
            └─ application_started=1 (p = interest × 0.4)
                 └─ application_completed=1 (p = interest × 0.7)
                      └─ product_activated=1 (p = interest × 0.8)
                           └─ first_transaction_done=1 (p = interest × 0.9)
```

Если `banner_shown=0` — все downstream-метрики обнуляются.

### Каскадная логика (voice)

```
call_connected=1 (p = voice_affinity × 0.7)
  └─ call_duration_sec: uniform(30, 900)
  └─ reached_argument_block=1 (p = interest × 0.8)
       └─ positive_reaction=1 (p = interest × 0.6)
            └─ verbal_agreement=1 (p = interest × 0.35)
                 └─ product_connected_after_call=1 (p = interest × 0.7)
```

Если `call_connected=0` — все voice-метрики равны 0.

---

## API

```http
POST /api/v1/metrics/render-prompt
{
  "classification": {...},
  "sales_argument": {...},
  "channel": "digital",
  "client_features": {...}
}
→ {"rendered_prompt": "...полный Jinja2-рендер..."}

POST /api/v1/metrics/generate
{
  "classification": {...},
  "sales_argument": {...},
  "channel": "digital",
  "client_features": {...},
  "method": "llm"   // или "random"
}
→ {
    "channel": "digital",
    "portrait": "P1",
    "interest_score": 0.72,
    "user_reaction_text": "...",
    "metrics": [...],
    "rendered_prompt": "...",
    "raw_llm_response": "..."
  }
```

---

## Батч-генерация

**Файл:** `pipeline/stage1_pipeline.py`

```python
from pipeline.stage1_pipeline import run_single, run_batch

# Один клиент с рандомными признаками
result = run_single(method="random")

# Батч из 100 клиентов
results = run_batch(n=100, method="random")

# Конкретный клиент через Mistral
result = run_single(
    client_features={"smb_type_code": "2", "okved_major_wrapped": "retail_trade"},
    interaction_type="banner",
    channel="digital",
    method="llm",
)
```

CLI:
```bash
python -m pipeline.stage1_pipeline
# → 5 случайных клиентов, метод random, вывод в stdout
```

Структура результата `run_single()`:
```python
{
    "client_features": {...},      # 8 редактируемых признаков
    "classification": {...},       # результат predict()
    "interaction_type": "banner",
    "sales_argument": {...},       # mock-аргумент
    "channel": "digital",
    "metrics_result": {
        "interest_score": 0.72,
        "user_reaction_text": "...",
        "metrics": [...]
    }
}
```
