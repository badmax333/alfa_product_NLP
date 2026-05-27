"""Pydantic-схемы API ступени 1."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Переопределения основных признаков; остальные берутся из DEFAULT_FEATURES."""

    smb_type_code: str | None = None
    okved_major_wrapped: str | None = None
    okved_major: str | None = None
    main_okved: str | None = None
    sourceattr_ccode: str | None = None
    days_from_ogrn: float | None = None
    week_sum_transactions: float | None = None
    categ_name: str | None = None

    @field_validator("smb_type_code", "okved_major", "main_okved", mode="before")
    @classmethod
    def coerce_code_fields(cls, v: Any) -> str | None:
        if v is None:
            return None
        return str(v)

    def to_overrides(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True)
        return {k: v for k, v in data.items() if v != ""}


class ShapFeatureItem(BaseModel):
    rank: int
    feature: str
    value: Any
    shap: float
    direction: str
    description: str


class ProductRecommendation(BaseModel):
    ame: int | None
    name: str


class PredictResponse(BaseModel):
    predicted_class: str
    class_description: str
    confidence: float
    probabilities: dict[str, float]
    recommended_product: ProductRecommendation
    top5_feature_importance: list[ShapFeatureItem]


class PresetInfo(BaseModel):
    id: str
    title: str
    description: str
    overrides: dict[str, Any]


class ConfigResponse(BaseModel):
    editable_features: list[str]
    feature_labels: dict[str, str]
    field_options: dict[str, list[dict[str, str]]]
    presets: list[PresetInfo]
    default_overrides: dict[str, Any] = Field(
        description="Значения редактируемых полей из базового профиля"
    )
    class_descriptions: dict[str, str] = Field(
        description="Описания 8 классов (портретов) для подсказок"
    )


# ---------------------------------------------------------------------------
# Sales arguments
# ---------------------------------------------------------------------------

class SalesArgumentItem(BaseModel):
    id: str
    interaction_type: str
    channel: str
    portrait: str
    portrait_label: str
    product_ame: int
    product_name: str
    headline: str
    body: str
    cta: str
    note: str


class InteractionTypeItem(BaseModel):
    id: str
    label: str
    channel: str
    description: str


class SalesArgumentsConfig(BaseModel):
    interaction_types: list[InteractionTypeItem]
    mock_arguments: list[SalesArgumentItem]


# ---------------------------------------------------------------------------
# Prompt rendering (для отображения реальных промптов в UI)
# ---------------------------------------------------------------------------

class RenderSalesArgPromptRequest(BaseModel):
    classification: dict[str, Any] = Field(description="Результат /api/v1/predict")
    interaction_type: str = Field(description="'banner' | 'push' | 'voice'")
    client_features: dict[str, Any] = Field(default_factory=dict)


class RenderMetricsPromptRequest(BaseModel):
    classification: dict[str, Any] = Field(description="Результат /api/v1/predict")
    sales_argument: dict[str, Any] = Field(description="Выбранный mock sales-аргумент")
    channel: str = Field(description="'digital' или 'voice'")
    client_features: dict[str, Any] = Field(default_factory=dict)


class RenderedPromptResponse(BaseModel):
    rendered_prompt: str


# ---------------------------------------------------------------------------
# Metrics generation
# ---------------------------------------------------------------------------

class GenerateMetricsRequest(BaseModel):
    classification: dict[str, Any] = Field(description="Результат /api/v1/predict")
    sales_argument: dict[str, Any] = Field(description="Выбранный mock sales-аргумент")
    channel: str = Field(description="'digital' или 'voice'")
    client_features: dict[str, Any] = Field(
        default_factory=dict,
        description="Признаки клиента из формы (8 редактируемых полей)",
    )
    method: str = Field(default="llm", description="'llm' — через Mistral, 'random' — локально")


class MetricValueItem(BaseModel):
    name: str
    label: str
    level: int
    level_name: str
    type: str
    unit: str
    description: str
    value: Any


class MetricsResponse(BaseModel):
    channel: str
    portrait: str
    rendered_prompt: str
    interest_score: float
    user_reaction_text: str
    metrics: list[MetricValueItem]
    raw_llm_response: str
