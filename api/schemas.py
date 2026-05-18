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
