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
    product_ame: int | None
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


class GenerateSalesArgumentRequest(RenderSalesArgPromptRequest):
    pass


class SalesArgumentResponse(SalesArgumentItem):
    rendered_prompt: str
    raw_llm_response: str


class RenderMetricsPromptRequest(BaseModel):
    classification: dict[str, Any] = Field(description="Результат /api/v1/predict")
    sales_argument: dict[str, Any] = Field(description="Выбранный sales-аргумент")
    channel: str = Field(description="'digital' или 'voice'")
    client_features: dict[str, Any] = Field(default_factory=dict)


class RenderedPromptResponse(BaseModel):
    rendered_prompt: str


# ---------------------------------------------------------------------------
# Metrics generation
# ---------------------------------------------------------------------------


class GenerateMetricsRequest(BaseModel):
    classification: dict[str, Any] = Field(description="Результат /api/v1/predict")
    sales_argument: dict[str, Any] = Field(description="Выбранный sales-аргумент")
    channel: str = Field(description="'digital' или 'voice'")
    client_features: dict[str, Any] = Field(
        default_factory=dict,
        description="Признаки клиента из формы (8 редактируемых полей)",
    )
    method: str = Field(
        default="llm", description="'llm' — через Mistral, 'random' — локально"
    )


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


# ---------------------------------------------------------------------------
# Product propensity scoring (Stage 2)
# ---------------------------------------------------------------------------


class PropensityFactorItem(BaseModel):
    feature: str
    label: str
    value: Any
    impact: float
    direction: str
    reason: str


class PropensityProductItem(BaseModel):
    rank: int
    product_id: str
    product_name: str
    product_ame: int | None
    scenario_id: int | None
    description: str
    anchor: bool
    propensity_score: float
    model_logit: float
    top_factors: list[PropensityFactorItem]


class PropensityScoreRequest(BaseModel):
    classification: dict[str, Any] = Field(description="Результат /api/v1/predict")
    client_features: dict[str, Any] = Field(default_factory=dict)
    metrics_result: dict[str, Any] = Field(
        description="Результат /api/v1/metrics/generate"
    )
    sales_argument: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=3, ge=1, le=10)


class RenderPropensityFeaturePromptRequest(BaseModel):
    classification: dict[str, Any] = Field(description="Результат /api/v1/predict")
    client_features: dict[str, Any] = Field(default_factory=dict)
    metrics_result: dict[str, Any] = Field(description="Результат /api/v1/metrics/generate")
    sales_argument: dict[str, Any] = Field(default_factory=dict)


class PropensityScoreResponse(BaseModel):
    portrait: str
    portrait_label: str
    model_source: str
    feature_source: str
    generated_features: dict[str, Any]
    feature_generation_reasoning: str
    feature_generation_prompt: str
    feature_generation_system_prompt: str
    feature_generation_raw_response: str
    interaction_interest_score: float | None
    top_products: list[PropensityProductItem]
    all_products: list[PropensityProductItem]


# ---------------------------------------------------------------------------
# Stage 2 sales argument generation
# ---------------------------------------------------------------------------


class RenderStage2SalesArgPromptRequest(BaseModel):
    classification: dict[str, Any] = Field(description="Результат /api/v1/predict")
    interaction_type: str = Field(description="'banner' | 'push' | 'voice'")
    client_features: dict[str, Any] = Field(default_factory=dict)
    propensity_product: dict[str, Any] = Field(
        description="Один продукт из top_products результата /api/v1/propensity/score"
    )
    stage1_argument: dict[str, Any] | None = Field(
        default=None,
        description="Аргумент Ступени 1 (headline, body, cta, product_name)",
    )
    stage1_metrics: dict[str, Any] | None = Field(
        default=None,
        description="Метрики взаимодействия Ступени 1 (interest_score, user_reaction_text)",
    )


class GenerateStage2SalesArgumentRequest(RenderStage2SalesArgPromptRequest):
    pass


class Stage2SalesArgumentResponse(SalesArgumentResponse):
    propensity_score: float | None = None


# ---------------------------------------------------------------------------
# Recycle / next onboarding cycle
# ---------------------------------------------------------------------------


class RecycleOnboardingRequest(BaseModel):
    classification: dict[str, Any] = Field(description="Результат /api/v1/predict")
    client_features: dict[str, Any] = Field(default_factory=dict)
    stage1_argument: dict[str, Any] = Field(description="Sales-аргумент Stage 1")
    stage1_metrics: dict[str, Any] = Field(description="Метрики Stage 1")
    propensity_result: dict[str, Any] = Field(description="Результат скоринга Stage 2")
    stage2_argument: dict[str, Any] = Field(description="Sales-аргумент Stage 2")
    stage2_metrics: dict[str, Any] = Field(description="Метрики Stage 2")
    selected_stage2_product: dict[str, Any] = Field(default_factory=dict)
    interaction_type: str = Field(default="banner", description="'banner' | 'push' | 'voice'")
    top_k: int = Field(default=3, ge=1, le=10)


class RecycleOnboardingResponse(BaseModel):
    cycle_number: int
    activation_status: str
    shown_product_ids: list[str]
    generated_features: dict[str, Any]
    feature_generation_reasoning: str
    feature_generation_prompt: str
    feature_generation_system_prompt: str
    propensity_model_source: str
    selected_product: PropensityProductItem
    next_argument: Stage2SalesArgumentResponse
    top_products: list[PropensityProductItem]
    all_products: list[PropensityProductItem]
    rendered_prompt: str
    system_prompt: str
    raw_llm_response: str
