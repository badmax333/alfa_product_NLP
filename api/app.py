"""FastAPI: демо ступени 1 — классификация бизнес-портрета."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.schemas import ConfigResponse, PredictRequest, PredictResponse, PresetInfo, ProductRecommendation, ShapFeatureItem
from config.stage1 import (
    DEFAULT_FEATURES,
    DEMO_PRESETS,
    EDITABLE_FEATURES,
    FEATURE_LABELS,
    FIELD_OPTIONS,
)
from models.classifier import predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"

app = FastAPI(
    title="Alfa Smart Onboarding — Ступень 1",
    description="Демо классификатора бизнес-портретов (CatBoost)",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/v1/config", response_model=ConfigResponse)
async def get_config():
    default_overrides = {k: DEFAULT_FEATURES[k] for k in EDITABLE_FEATURES}
    presets = [PresetInfo(**p) for p in DEMO_PRESETS]
    return ConfigResponse(
        editable_features=EDITABLE_FEATURES,
        feature_labels=FEATURE_LABELS,
        field_options=FIELD_OPTIONS,
        presets=presets,
        default_overrides=default_overrides,
    )


@app.post("/api/v1/predict", response_model=PredictResponse)
async def predict_segment(body: PredictRequest):
    result = predict(body.to_overrides())
    return PredictResponse(
        predicted_class=result["predicted_class"],
        class_description=result["class_description"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        recommended_product=ProductRecommendation(**result["recommended_product"]),
        top5_feature_importance=[ShapFeatureItem(**item) for item in result["top5_feature_importance"]],
    )
