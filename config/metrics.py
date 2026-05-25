"""Определение метрик оценки взаимодействия пользователя с первым предложением."""

from typing import Any

# ---------------------------------------------------------------------------
# Структура одной метрики
# ---------------------------------------------------------------------------
# name          — технический идентификатор
# label         — человекочитаемое название
# level         — уровень воронки (1–5)
# level_name    — название уровня
# channel       — "digital" | "voice" | "both"
# type          — "binary" | "integer" | "float" | "duration_sec" | "enum"
# range         — [min, max] или список допустимых значений для enum
# unit          — единица измерения (для отображения)
# description   — что измеряет метрика
# ---------------------------------------------------------------------------

LEVEL_NAMES = {
    1: "Доставка / контакт",
    2: "Внимание / интерес",
    3: "Целевое действие",
    4: "Активация / использование",
    5: "Качество / негатив",
}

DIGITAL_METRICS: list[dict[str, Any]] = [
    # --- Уровень 1: Доставка ---
    {
        "name": "banner_shown",
        "label": "Баннер показан",
        "level": 1,
        "level_name": LEVEL_NAMES[1],
        "channel": "digital",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Баннер/сообщение был отображён в интерфейсе клиента.",
    },
    {
        "name": "banner_visible_sec",
        "label": "Время видимости баннера (сек)",
        "level": 1,
        "level_name": LEVEL_NAMES[1],
        "channel": "digital",
        "type": "duration_sec",
        "range": [0, 120],
        "unit": "сек",
        "description": "Сколько секунд баннер находился в зоне видимости экрана клиента.",
    },
    {
        "name": "push_delivered",
        "label": "Push/Email доставлен",
        "level": 1,
        "level_name": LEVEL_NAMES[1],
        "channel": "digital",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Push-уведомление или email были успешно доставлены клиенту.",
    },
    {
        "name": "push_opened",
        "label": "Push/Email открыт",
        "level": 1,
        "level_name": LEVEL_NAMES[1],
        "channel": "digital",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент открыл push-уведомление или email.",
    },
    # --- Уровень 2: Внимание / интерес ---
    {
        "name": "banner_clicked",
        "label": "Клик по баннеру",
        "level": 2,
        "level_name": LEVEL_NAMES[2],
        "channel": "digital",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент нажал на баннер или кнопку CTA.",
    },
    {
        "name": "product_page_visited",
        "label": "Переход в раздел продукта",
        "level": 2,
        "level_name": LEVEL_NAMES[2],
        "channel": "digital",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент перешёл на страницу с подробным описанием предлагаемого продукта.",
    },
    {
        "name": "time_on_product_page_sec",
        "label": "Время на странице продукта (сек)",
        "level": 2,
        "level_name": LEVEL_NAMES[2],
        "channel": "digital",
        "type": "duration_sec",
        "range": [0, 600],
        "unit": "сек",
        "description": "Сколько секунд клиент провёл на странице продукта.",
    },
    {
        "name": "scroll_depth_pct",
        "label": "Глубина прокрутки карточки (%)",
        "level": 2,
        "level_name": LEVEL_NAMES[2],
        "channel": "digital",
        "type": "float",
        "range": [0.0, 100.0],
        "unit": "%",
        "description": "На сколько процентов клиент прокрутил карточку предложения.",
    },
    {
        "name": "remind_later_clicked",
        "label": "«Напомнить позже» нажато",
        "level": 2,
        "level_name": LEVEL_NAMES[2],
        "channel": "digital",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент выбрал опцию «Напомнить позже» — проявил интерес, но не готов сейчас.",
    },
    {
        "name": "repeated_view_count",
        "label": "Повторных просмотров",
        "level": 2,
        "level_name": LEVEL_NAMES[2],
        "channel": "digital",
        "type": "integer",
        "range": [0, 10],
        "unit": "раз",
        "description": "Сколько раз клиент возвращался к карточке предложения.",
    },
    # --- Уровень 3: Целевое действие ---
    {
        "name": "application_started",
        "label": "Начало заявки",
        "level": 3,
        "level_name": LEVEL_NAMES[3],
        "channel": "digital",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент начал заполнять заявку на подключение продукта.",
    },
    {
        "name": "application_completed",
        "label": "Заявка оформлена",
        "level": 3,
        "level_name": LEVEL_NAMES[3],
        "channel": "digital",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент завершил и отправил заявку на подключение продукта.",
    },
    {
        "name": "time_to_action_min",
        "label": "Время от показа до действия (мин)",
        "level": 3,
        "level_name": LEVEL_NAMES[3],
        "channel": "digital",
        "type": "float",
        "range": [0.0, 10080.0],
        "unit": "мин",
        "description": "Сколько минут прошло от первого показа до совершения целевого действия.",
    },
    # --- Уровень 4: Активация / использование ---
    {
        "name": "product_activated",
        "label": "Продукт активирован",
        "level": 4,
        "level_name": LEVEL_NAMES[4],
        "channel": "digital",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Продукт был фактически подключён клиенту.",
    },
    {
        "name": "first_transaction_done",
        "label": "Первая транзакция совершена",
        "level": 4,
        "level_name": LEVEL_NAMES[4],
        "channel": "digital",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент совершил первую операцию через подключённый продукт.",
    },
    {
        "name": "days_to_first_use",
        "label": "Дней до первого использования",
        "level": 4,
        "level_name": LEVEL_NAMES[4],
        "channel": "digital",
        "type": "integer",
        "range": [0, 90],
        "unit": "дней",
        "description": "Сколько дней прошло от активации до первого реального использования продукта.",
    },
    # --- Уровень 5: Качество / негатив ---
    {
        "name": "banner_dismissed",
        "label": "Баннер скрыт",
        "level": 5,
        "level_name": LEVEL_NAMES[5],
        "channel": "digital",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент намеренно закрыл или скрыл баннер.",
    },
    {
        "name": "push_unsubscribed",
        "label": "Отписка от push/email",
        "level": 5,
        "level_name": LEVEL_NAMES[5],
        "channel": "digital",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент отписался от push-уведомлений или email-рассылки.",
    },
    {
        "name": "complaint_filed",
        "label": "Жалоба подана",
        "level": 5,
        "level_name": LEVEL_NAMES[5],
        "channel": "digital",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент оставил жалобу на навязчивость или нерелевантность предложения.",
    },
    {
        "name": "negative_rating",
        "label": "Оценка предложения (1–5)",
        "level": 5,
        "level_name": LEVEL_NAMES[5],
        "channel": "digital",
        "type": "integer",
        "range": [1, 5],
        "unit": "баллов",
        "description": "Оценка, которую клиент поставил предложению в интерфейсе (1 — очень плохо, 5 — отлично).",
    },
]

VOICE_METRICS: list[dict[str, Any]] = [
    # --- Уровень 1: Доставка / контакт ---
    {
        "name": "call_connected",
        "label": "Дозвон состоялся",
        "level": 1,
        "level_name": LEVEL_NAMES[1],
        "channel": "voice",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Оператор дозвонился до клиента и разговор был принят.",
    },
    {
        "name": "call_duration_sec",
        "label": "Длительность разговора (сек)",
        "level": 1,
        "level_name": LEVEL_NAMES[1],
        "channel": "voice",
        "type": "duration_sec",
        "range": [0, 1800],
        "unit": "сек",
        "description": "Суммарная длительность состоявшегося звонка.",
    },
    {
        "name": "reached_argument_block",
        "label": "Оператор дошёл до блока с аргументом",
        "level": 1,
        "level_name": LEVEL_NAMES[1],
        "channel": "voice",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Разговор продолжался достаточно долго, чтобы оператор успел озвучить sales-аргумент.",
    },
    {
        "name": "client_didnt_hangup_before_argument",
        "label": "Клиент не прервал разговор до аргумента",
        "level": 1,
        "level_name": LEVEL_NAMES[1],
        "channel": "voice",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент не положил трубку до того, как оператор озвучил предложение.",
    },
    # --- Уровень 2: Внимание / интерес ---
    {
        "name": "positive_reaction",
        "label": "Позитивная / нейтрально-заинтересованная реакция",
        "level": 2,
        "level_name": LEVEL_NAMES[2],
        "channel": "voice",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент отреагировал позитивно или с интересом, а не отказал сразу.",
    },
    {
        "name": "clarifying_questions_count",
        "label": "Уточняющих вопросов задано",
        "level": 2,
        "level_name": LEVEL_NAMES[2],
        "channel": "voice",
        "type": "integer",
        "range": [0, 10],
        "unit": "шт",
        "description": "Сколько уточняющих вопросов задал клиент в ходе разговора — прокси интереса.",
    },
    {
        "name": "requested_link_or_materials",
        "label": "Просьба прислать ссылку / материалы",
        "level": 2,
        "level_name": LEVEL_NAMES[2],
        "channel": "voice",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент попросил прислать ссылку или дополнительные материалы для ознакомления.",
    },
    {
        "name": "agreed_to_callback",
        "label": "Согласился на перезвон",
        "level": 2,
        "level_name": LEVEL_NAMES[2],
        "channel": "voice",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент дал согласие на повторный звонок для продолжения разговора.",
    },
    # --- Уровень 3: Целевое действие ---
    {
        "name": "verbal_agreement",
        "label": "Устное согласие на подключение",
        "level": 3,
        "level_name": LEVEL_NAMES[3],
        "channel": "voice",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент дал устное согласие на подключение предлагаемого продукта.",
    },
    {
        "name": "followed_link_after_call",
        "label": "Перешёл по ссылке после звонка",
        "level": 3,
        "level_name": LEVEL_NAMES[3],
        "channel": "voice",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "После разговора клиент перешёл по отправленной ссылке в приложение или на сайт.",
    },
    # --- Уровень 4: Активация / использование ---
    {
        "name": "product_connected_after_call",
        "label": "Продукт подключён после звонка",
        "level": 4,
        "level_name": LEVEL_NAMES[4],
        "channel": "voice",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент фактически подключил продукт в течение 7 дней после звонка.",
    },
    {
        "name": "first_operation_after_call",
        "label": "Первая операция после звонка",
        "level": 4,
        "level_name": LEVEL_NAMES[4],
        "channel": "voice",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент совершил первую транзакцию через подключённый продукт.",
    },
    # --- Уровень 5: Качество / негатив ---
    {
        "name": "negative_reaction_voice",
        "label": "Негативная реакция / раздражение",
        "level": 5,
        "level_name": LEVEL_NAMES[5],
        "channel": "voice",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент явно выразил раздражение, агрессию или недовольство звонком.",
    },
    {
        "name": "complaint_after_call",
        "label": "Жалоба после звонка",
        "level": 5,
        "level_name": LEVEL_NAMES[5],
        "channel": "voice",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент оставил официальную жалобу после звонка.",
    },
    {
        "name": "requested_no_more_calls",
        "label": "Просьба больше не звонить",
        "level": 5,
        "level_name": LEVEL_NAMES[5],
        "channel": "voice",
        "type": "binary",
        "range": [0, 1],
        "unit": "bool",
        "description": "Клиент явно попросил не беспокоить его звонками.",
    },
]

ALL_METRICS: list[dict[str, Any]] = DIGITAL_METRICS + VOICE_METRICS


def get_metrics_for_channel(channel: str) -> list[dict[str, Any]]:
    """Вернуть метрики для заданного канала."""
    if channel == "digital":
        return DIGITAL_METRICS
    if channel == "voice":
        return VOICE_METRICS
    return ALL_METRICS


# Характеристики поведения по портретам (используются в промпте)
PORTRAIT_BEHAVIORAL_PROFILES: dict[str, dict] = {
    "P1": {
        "name": "Розничный продавец",
        "digital_affinity": "medium",
        "voice_affinity": "high",
        "interest_base": 0.72,
        "typical_behavior": (
            "Владельцы розничных точек чаще взаимодействуют через мобильное приложение "
            "между рабочими сменами. Хорошо реагируют на аргументы про скорость расчётов "
            "и экономию на комиссии. Предложение торгового эквайринга воспринимают как "
            "естественную необходимость. Клики по баннеру выше средних, но оформление "
            "заявки часто откладывают на вечер."
        ),
        "negative_triggers": "давление, сложная форма заявки, страх нового оборудования",
    },
    "P2": {
        "name": "IT / Онлайн-бизнес",
        "digital_affinity": "very_high",
        "voice_affinity": "low",
        "interest_base": 0.65,
        "typical_behavior": (
            "IT-предприниматели критически оценивают банковские продукты. "
            "Высокая вовлечённость в digital-канале: читают условия, сравнивают тарифы. "
            "Звонки воспринимают как вторжение. Глубина прокрутки карточки высокая, "
            "но конверсия в действие ниже средней — долго взвешивают решение."
        ),
        "negative_triggers": "неточные условия, высокая комиссия, назойливые звонки",
    },
    "P3": {
        "name": "Строитель / Подрядчик",
        "digital_affinity": "low",
        "voice_affinity": "very_high",
        "interest_base": 0.68,
        "typical_behavior": (
            "Предпочитают голосовое общение. В цифровом канале практически не взаимодействуют "
            "с баннерами. Зарплатный проект воспринимают как облегчение административной нагрузки. "
            "При дозвоне задают конкретные вопросы про сроки и документы, конверсия в устное "
            "согласие выше среднего, если оператор говорит коротко и по делу."
        ),
        "negative_triggers": "длинное объяснение, сложная бюрократия, недоверие к банку",
    },
    "P4": {
        "name": "Фрилансер / Самозанятый ИП",
        "digital_affinity": "high",
        "voice_affinity": "low",
        "interest_base": 0.55,
        "typical_behavior": (
            "Стоимость важнее всего. Быстро принимают решения в digital-канале. "
            "Подписки воспринимают как нежелательные расходы, поэтому аргумент должен "
            "показывать прямую выгоду. CTR по баннерам средний, но если кликнули — "
            "высокая вероятность оформления заявки в тот же день."
        ),
        "negative_triggers": "скрытые комиссии, сложный онбординг, ощущение навязывания",
    },
    "P5": {
        "name": "Новое ЮЛ / Стартап",
        "digital_affinity": "high",
        "voice_affinity": "medium",
        "interest_base": 0.60,
        "typical_behavior": (
            "Активно изучают продукты, пробуют всё новое. CTR высокий, но конверсия "
            "в активацию низкая — нет понимания, какой продукт нужен первым. "
            "Хорошо реагируют на образовательный контент и пошаговые инструкции."
        ),
        "negative_triggers": "непонятные условия, отсутствие поддержки, страх первой ошибки",
    },
    "P6": {
        "name": "Услуги / HoReCa",
        "digital_affinity": "medium",
        "voice_affinity": "medium",
        "interest_base": 0.70,
        "typical_behavior": (
            "Постоянно заняты в рабочее время. CTR в digital ниже среднего из-за нехватки времени, "
            "но если видят баннер вечером — конверсия выше. По телефону реагируют живо, "
            "особенно на аргументы про скорость расчётов с гостями."
        ),
        "negative_triggers": "звонки в обеденный час, длинные скрипты, неудобное оборудование",
    },
    "P7": {
        "name": "Логистика / Транспорт",
        "digital_affinity": "medium",
        "voice_affinity": "high",
        "interest_base": 0.66,
        "typical_behavior": (
            "Прагматики. Зарплатный проект воспринимают как решение реальной проблемы с выплатой "
            "водителям. В digital взаимодействуют редко. По телефону разговор идёт по делу, "
            "мало вопросов, конверсия в согласие средняя, зависит от загруженности."
        ),
        "negative_triggers": "сложный документооборот, задержки подключения, неудобный интерфейс",
    },
    "P8": {
        "name": "Агробизнес / КФХ",
        "digital_affinity": "low",
        "voice_affinity": "very_high",
        "interest_base": 0.62,
        "typical_behavior": (
            "Консервативны, не доверяют новым продуктам. Предпочитают общаться по телефону "
            "или лично в отделении. Digital-активность низкая. Зарплатный проект принимают "
            "при наличии конкретного примера и рекомендации. Сезонность влияет на готовность: "
            "в уборочный сезон разговор невозможен."
        ),
        "negative_triggers": "недоверие к технологиям, страх перемен, сезонная занятость",
    },
}
