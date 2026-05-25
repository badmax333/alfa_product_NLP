# Классификатор бизнес-портретов (Ступень 1)

## Модель

**Алгоритм:** CatBoost Multiclass  
**Файл модели:** `notebooks/alfa_classifier.cbm`  
**Конфигурация:** `config/stage1.py`  
**Инференс + SHAP:** `models/classifier.py`

```python
CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=3.0,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    early_stopping_rounds=80,
    auto_class_weights="Balanced",
)
```

**Результаты (синтетический датасет, 5000 строк):**
- Accuracy (val): 0.999 · F1-macro: 0.999
- На реальных данных ожидаемо: Accuracy 0.65–0.80, F1-macro 0.60–0.75

## 8 бизнес-портретов

| ID | Портрет | Якорный продукт | Ключевые признаки |
|---|---|---|---|
| P1 | 🏪 Розничный продавец | Торговый эквайринг (AME-7) | ОКВЭД 47, ИП/ЮЛ, branch |
| P2 | 💻 IT / Онлайн-бизнес | Интернет-эквайринг (AME-11) | ОКВЭД 62–63, online |
| P3 | 🏗️ Строитель / Подрядчик | Зарплатный проект (AME-10) | ОКВЭД 41–43, ЮЛ, высокий оборот |
| P4 | 🎓 Фрилансер / Самозанятый ИП | Подписки платежей (AME-6) | ИП, ОКВЭД 62/73/74, низкий оборот |
| P5 | 🌱 Новое ЮЛ / Стартап | Знакомство с тарифом (AME-2) | возраст < 90 дней |
| P6 | 🍽️ Услуги / HoReCa | Торговый эквайринг (AME-7) | ОКВЭД 55–56, ИП/ЮЛ |
| P7 | 🚚 Логистика / Транспорт | Зарплатный проект (AME-10) | ОКВЭД 49–52, ЮЛ/ИП |
| P8 | 🌾 Агробизнес / КФХ | Зарплатный проект (AME-10) | КФХ, ОКВЭД 01–02/10 |

## 50 признаков

Полный список в `config/stage1.py` → `FEATURE_COLS`. Разбивка по группам:

### Регистрационные (11)
```
smb_type_code          # ЮЛ=1, ИП=2, КФХ=3
main_okved             # Основной ОКВЭД
okved_major            # Первые 2 знака ОКВЭД
okved_major_wrapped    # Укрупнённая отрасль: retail_trade, it_software, construction, agriculture…
okved_cnt_total        # Всего ОКВЭД
okved_groups_unique_cnt
okved_groups_share
days_from_ogrn         # Возраст бизнеса (дней)
days_from_smb          # Дней в реестре МСП
```

### Временные (6)
```
ogrn_days_end_month    # Дней до конца месяца
ogrn_days_end_quarter
week_sum_transactions  # Сумма транзакций за неделю (₽)
week_mean_transactions # Количество транзакций за неделю
share_last_month       # % от недельного оборота
share_last_3_months
```

### Продуктовые (5)
```
acquiring_num_live     # Активных эквайринговых продуктов
zpp_num_live           # Активных зарплатных проектов
nkop_num_live          # Активных «Налоговых копилок»
rko_num_live           # Активных расчётных счетов
srvpackage_sale_uk     # Пакет услуг на старте
```

### Географические (9)
```
city, addrf_region_name, postindex_town_code
postindex_area_code, oktmo_reg_code, oktmo_ccode
oktmo_municipality_code, kpp_region_code, addrf_region_code
```

### Канальные и административные (14)
```
sourceattr_ccode      # Канал привлечения: branch, online, referral…
branch_eq_ccode, sks_city, division_name
regorg_code, registration_reason_code
pensfund_authority_code
categ_name            # Категория расходов: grocery, fuel, software…
```

### Директорские / собственнические (5)
```
apin_salary_last_days
apin_product_active_days
xpin_start_days
xpin_birth_days
days_from_authperson_registration
```

## 8 редактируемых признаков в UI

В демо-интерфейсе пользователь может изменить 8 ключевых признаков (`config/stage1.py` → `EDITABLE_FEATURES`):

```python
EDITABLE_FEATURES = [
    "smb_type_code",       # Тип субъекта (ЮЛ / ИП / КФХ)
    "okved_major_wrapped", # Укрупнённая отрасль
    "okved_major",         # ОКВЭД (2 знака)
    "main_okved",          # Основной ОКВЭД
    "sourceattr_ccode",    # Канал привлечения
    "days_from_ogrn",      # Возраст бизнеса (дней)
    "week_sum_transactions",# Оборот за неделю (₽)
    "categ_name",          # Категория расходов
]
```

Остальные 42 признака фиксируются значениями из `DEFAULT_FEATURES` (базовый профиль: розничная ИП, Краснодар, 400 дней, 130k ₽/неделя).

## Per-sample SHAP

На каждом инференсе модель возвращает топ-5 признаков, повлиявших на решение. Используется для:
- Передачи контекста в LLM при генерации sales-аргумента
- Отображения в UI (Tab 1 и Tab 2)
- Аудита решений

```python
# models/classifier.py
shap_vals = model.get_feature_importance(
    data=pool,
    type="ShapValues",
    shap_calc_type="Regular",
)
# shape: (n_samples, n_features + 1, n_classes)
# Топ-5 по |shap| для предсказанного класса
```

Пример вывода:
```json
[
  {"rank": 1, "feature": "okved_major_wrapped", "value": "retail_trade", "shap": 0.0101, "direction": "▲"},
  {"rank": 2, "feature": "smb_type_code",       "value": "2",            "shap": 0.0050, "direction": "▲"},
  {"rank": 3, "feature": "categ_name",           "value": "grocery",      "shap": 0.0038, "direction": "▲"},
  {"rank": 4, "feature": "week_sum_transactions","value": "130000",       "shap": 0.0028, "direction": "▲"},
  {"rank": 5, "feature": "days_from_ogrn",       "value": "400",          "shap": 0.0015, "direction": "▲"}
]
```

## API

```http
POST /api/v1/predict
Content-Type: application/json

{
  "smb_type_code": "2",
  "okved_major_wrapped": "retail_trade",
  "okved_major": "47",
  "main_okved": "47",
  "sourceattr_ccode": "branch",
  "days_from_ogrn": 400,
  "week_sum_transactions": 130000,
  "categ_name": "grocery"
}
```

```json
{
  "predicted_class": "P1",
  "class_description": "🏪 Розничный продавец",
  "confidence": 0.829,
  "probabilities": {"P1": 0.829, "P2": 0.051, ...},
  "recommended_product": {"ame": 7, "name": "Торговый эквайринг"},
  "top5_feature_importance": [...]
}
```
