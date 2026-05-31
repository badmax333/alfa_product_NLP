# Sales-аргументы (Ступень 1)

## Типы взаимодействия

Определены в `config/sales_arguments.py` → `INTERACTION_TYPES`:

| ID | Формат | Канал | Описание |
|---|---|---|---|
| `banner` | Баннер в приложении | digital | Визуальная карточка с заголовком, телом и CTA-кнопкой |
| `push` | Push-уведомление / SMS | digital | Короткое сообщение, ~100 символов |
| `voice` | Голосовой скрипт | voice | Скрипт для оператора: открытие → аргумент → возражение → закрытие |

## Демо-примеры аргументов

Три готовых примера остаются в `config/sales_arguments.py` → `MOCK_SALES_ARGUMENTS` для совместимости и локальных сценариев:

| ID | Тип | Портрет | Продукт |
|---|---|---|---|
| `mock_banner_p1` | banner | P1 Розничный продавец | Торговый эквайринг (AME-7) |
| `mock_push_p2` | push | P2 IT / Онлайн-бизнес | Интернет-эквайринг (AME-11) |
| `mock_voice_p3` | voice | P3 Строитель / Подрядчик | Зарплатный проект (AME-10) |

Каждый аргумент содержит поля: `headline`, `body`, `cta`, `note` (пояснение, почему такой текст).

В UI эти примеры больше не выбираются как основной результат: Tab 2 генерирует персонализированный аргумент через Mistral.

## Генерация через LLM

> **Статус:** Реализовано через `services/sales_argument_generator.py` и `/api/v1/sales-args/generate`.

### Где лежат промпты

```
prompts/
├── stage1_sales_argument.j2        ← шаблон для генерации аргумента
└── stage1_metrics_generation.j2    ← шаблон для генерации метрик
```

### Что передаётся в LLM

```
Портрет клиента (P1–P8) + описание
Якорный продукт (AME-код, название)
Top-5 SHAP-признаков (объяснение решения классификатора)
Тип взаимодействия (banner / push / voice)
Поведенческий профиль портрета (из config/metrics.py)
```

### Ожидаемый выход LLM

```json
{
  "headline": "...",
  "body": "...",
  "cta": "..."
}
```

Формат зависит от типа: banner (2–3 предложения), push (~100 символов), voice (4 блока по структуре).

### Архитектура с compliance-checker

```
LLM #1 (Generator) → черновик аргумента
        │
        ▼
LLM #2 (Compliance) → проверка:
  ✗ Обещания гарантированной прибыли
  ✗ Гарантии доходности
  ✗ Некорректные сравнения с конкурентами
  ✗ Нарушения требований ЦБ РФ / ФАС
        │
    ┌───┴───┐
  PASS     FAIL → fallback-шаблон
    │
    ▼
  CRM / доставка
```

## Промпт в UI (Tab 2)

В интерфейсе Tab 2 отображается **реальный рендеренный Jinja2-промпт**, а кнопка генерации отправляет его в Mistral.

### Как это работает

1. При выборе типа взаимодействия JS делает POST-запрос на `/api/v1/sales-args/render-prompt`.
2. Backend вызывает `services/sales_arg_renderer.py` → `render_sales_arg_prompt()`.
3. Функция загружает `prompts/stage1_sales_argument.j2` через Jinja2 и подставляет данные клиента.
4. По кнопке генерации JS делает POST-запрос на `/api/v1/sales-args/generate`.
5. Backend вызывает `services/sales_argument_generator.py` → `generate_sales_argument()`, парсит JSON из ответа Mistral и возвращает карточку аргумента.

```python
# services/sales_arg_renderer.py
def render_sales_arg_prompt(
    classification: dict,
    interaction_type: str,
    client_features: dict,
) -> str:
    """Рендерит stage1_sales_argument.j2 с данными клиента."""
```

```http
POST /api/v1/sales-args/render-prompt
{
  "classification": {"predicted_class": "P1", ...},
  "interaction_type": "banner",
  "client_features": {"smb_type_code": "2", ...}
}
→ {"rendered_prompt": "...полный текст промпта..."}
```

```http
POST /api/v1/sales-args/generate
{
  "classification": {"predicted_class": "P1", ...},
  "interaction_type": "banner",
  "client_features": {"smb_type_code": "2", ...}
}
→ {
  "id": "llm_banner_p1",
  "headline": "...",
  "body": "...",
  "cta": "...",
  "rendered_prompt": "...",
  "raw_llm_response": "..."
}
```
