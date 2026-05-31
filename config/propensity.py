"""Конфигурация stage2: продукты и правила скоринга склонности."""

from typing import Any

PROPENSITY_PRODUCTS: dict[str, dict[str, Any]] = {
    "zpp": {
        "name": "Зарплатный проект",
        "ame": 10,
        "anchor": False,
        "scenario_id": 10,
        "description": "Выплаты сотрудникам и подрядчикам через зарплатный проект.",
    },
    "alfa_payments": {
        "name": "Подписка Альфа-платежи",
        "ame": 6,
        "anchor": True,
        "scenario_id": 6,
        "description": "Пакет платежей для активного расчетного бизнеса.",
    },
    "nachalo": {
        "name": "Подписка «Начало»",
        "ame": 6,
        "anchor": True,
        "scenario_id": 6,
        "description": "Стартовый пакет для молодых компаний и клиентов без пакета услуг.",
    },
    "trade_acquiring": {
        "name": "Торговый эквайринг",
        "ame": 7,
        "anchor": False,
        "scenario_id": 7,
        "description": "Прием оплат картами в офлайн-точках.",
    },
    "internet_acquiring": {
        "name": "Интернет-эквайринг",
        "ame": 11,
        "anchor": False,
        "scenario_id": 11,
        "description": "Прием онлайн-платежей на сайте, в приложении или через API.",
    },
    "tax_jar": {
        "name": "Налоговая копилка",
        "ame": 14,
        "anchor": True,
        "scenario_id": 14,
        "description": "Автоматическое резервирование денег под будущие налоги.",
    },
    "savings": {
        "name": "Накопительный счет",
        "ame": None,
        "anchor": True,
        "scenario_id": None,
        "description": "Размещение свободного остатка с сохранением ликвидности.",
    },
    "accounting": {
        "name": "Бухгалтерия",
        "ame": 12,
        "anchor": True,
        "scenario_id": 12,
        "description": "Помощь с бухгалтерией, отчетностью и налоговыми операциями.",
    },
}

PRODUCT_IDS = list(PROPENSITY_PRODUCTS)

SEGMENT_PRODUCT_BIAS: dict[str, dict[str, float]] = {
    "P1": {"zpp": 0.15, "accounting": 0.1},
    "P2": {"alfa_payments": 0.2, "internet_acquiring": 0.15},
    "P3": {"trade_acquiring": 0.2, "savings": 0.1},
    "P4": {"internet_acquiring": 0.2, "alfa_payments": 0.1},
    "P5": {"nachalo": 0.25, "tax_jar": 0.1},
    "P6": {"accounting": 0.2, "tax_jar": 0.15},
    "P7": {"trade_acquiring": 0.15, "zpp": 0.1},
    "P8": {"zpp": 0.1, "nachalo": 0.15},
}

RETAIL_OKVED = {47, 49, 52, 53, 55, 56}
DIGITAL_CATEGORIES = {"digital_services", "online_ads", "software", "saas", "electronics"}
RETAIL_CATEGORIES = {"fuel", "equipment", "electronics", "food_service", "healthcare", "grocery"}
TAX_CATEGORIES = {"tax_payment", "bank_operations", "misc"}

PROPENSITY_FEATURE_LABELS: dict[str, str] = {
    "priority_segment": "Портрет P1-P8",
    "smb_type_code": "Тип клиента",
    "okved_major": "ОКВЭД",
    "okved_major_wrapped": "Отраслевая группа",
    "categ_name": "Категория бизнеса",
    "days_from_ogrn": "Возраст бизнеса",
    "week_sum_transactions": "Недельный оборот",
    "week_mean_transactions": "Среднее число транзакций",
    "sourceattr_ccode": "Канал привлечения",
    "srvpackage_sale_uk": "Пакет услуг",
    "acquiring_num_live": "Подключенный эквайринг",
    "zpp_num_live": "Подключенный ЗПП",
    "nkop_num_live": "Подключенная налоговая копилка",
    "accum": "Накопительный профиль",
    "impnt": "Цифровая вовлеченность",
    "complexity": "Сложность профиля",
}
