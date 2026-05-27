"""FastAPI: демо ступени 1 — классификация бизнес-портрета."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

load_dotenv()
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.schemas import (
    ConfigResponse,
    GenerateSalesArgumentRequest,
    GenerateMetricsRequest,
    InteractionTypeItem,
    MetricValueItem,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    PresetInfo,
    ProductRecommendation,
    PropensityProductItem,
    PropensityScoreRequest,
    PropensityScoreResponse,
    RenderedPromptResponse,
    RenderMetricsPromptRequest,
    RenderSalesArgPromptRequest,
    SalesArgumentResponse,
    SalesArgumentItem,
    SalesArgumentsConfig,
    ShapFeatureItem,
)
from config.sales_arguments import INTERACTION_TYPES, MOCK_SALES_ARGUMENTS
from config.stage1 import (
    CLASS_DESCRIPTIONS,
    DEFAULT_FEATURES,
    DEMO_PRESETS,
    EDITABLE_FEATURES,
    FEATURE_LABELS,
    FIELD_OPTIONS,
)
from models.classifier import predict
from services.metrics_generator import generate_metrics, render_metrics_prompt
from services.propensity_scorer import score_propensity
from services.random_metrics_generator import generate_metrics_random
from services.sales_argument_generator import generate_sales_argument
from services.sales_arg_renderer import render_sales_arg_prompt

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
        class_descriptions=CLASS_DESCRIPTIONS,
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


@app.get("/api/v1/sales-args/config", response_model=SalesArgumentsConfig)
async def get_sales_args_config():
    """Вернуть типы взаимодействия и mock sales-аргументы."""
    return SalesArgumentsConfig(
        interaction_types=[InteractionTypeItem(**t) for t in INTERACTION_TYPES],
        mock_arguments=[SalesArgumentItem(**a) for a in MOCK_SALES_ARGUMENTS],
    )


@app.post("/api/v1/sales-args/render-prompt", response_model=RenderedPromptResponse)
async def render_sales_arg_prompt_endpoint(body: RenderSalesArgPromptRequest):
    """Рендерить шаблон sales-аргумента для отображения промпта в UI (Tab 2)."""
    try:
        prompt = render_sales_arg_prompt(
            classification=body.classification,
            interaction_type=body.interaction_type,
            client_features=body.client_features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка рендеринга промпта: {e}")
    return RenderedPromptResponse(rendered_prompt=prompt)


@app.post("/api/v1/sales-args/generate", response_model=SalesArgumentResponse)
async def generate_sales_arg_endpoint(body: GenerateSalesArgumentRequest):
    """Сгенерировать персонализированный sales-аргумент через Mistral."""
    try:
        result = generate_sales_argument(
            classification=body.classification,
            interaction_type=body.interaction_type,
            client_features=body.client_features,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации sales-аргумента: {e}")

    return SalesArgumentResponse(**result)


@app.post("/api/v1/metrics/render-prompt", response_model=RenderedPromptResponse)
async def render_metrics_prompt_endpoint(body: RenderMetricsPromptRequest):
    """Рендерить промпт для генерации метрик без вызова Mistral (для превью в UI)."""
    if body.channel not in ("digital", "voice"):
        raise HTTPException(status_code=422, detail="channel должен быть 'digital' или 'voice'")
    try:
        prompt = render_metrics_prompt(
            classification=body.classification,
            sales_argument=body.sales_argument,
            channel=body.channel,
            client_features=body.client_features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка рендеринга промпта: {e}")
    return RenderedPromptResponse(rendered_prompt=prompt)


@app.post("/api/v1/metrics/generate", response_model=MetricsResponse)
async def generate_metrics_endpoint(body: GenerateMetricsRequest):
    """Сгенерировать синтетические метрики взаимодействия (LLM или случайно)."""
    if body.channel not in ("digital", "voice"):
        raise HTTPException(status_code=422, detail="channel должен быть 'digital' или 'voice'")
    if body.method not in ("llm", "random"):
        raise HTTPException(status_code=422, detail="method должен быть 'llm' или 'random'")
    try:
        if body.method == "random":
            result = generate_metrics_random(
                classification=body.classification,
                sales_argument=body.sales_argument,
                channel=body.channel,
                client_features=body.client_features,
            )
        else:
            result = generate_metrics(
                classification=body.classification,
                sales_argument=body.sales_argument,
                channel=body.channel,
                client_features=body.client_features,
            )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {e}")

    return MetricsResponse(
        channel=result["channel"],
        portrait=result["portrait"],
        rendered_prompt=result["rendered_prompt"],
        interest_score=result["interest_score"],
        user_reaction_text=result["user_reaction_text"],
        metrics=[MetricValueItem(**m) for m in result["metrics"]],
        raw_llm_response=result["raw_llm_response"],
    )


@app.post("/api/v1/propensity/score", response_model=PropensityScoreResponse)
async def score_propensity_endpoint(body: PropensityScoreRequest):
    """Оценить склонность клиента к продуктам после расчета метрик взаимодействия."""
    try:
        result = score_propensity(
            classification=body.classification,
            client_features=body.client_features,
            metrics_result=body.metrics_result,
            top_k=body.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка скоринга склонности: {e}")

    return PropensityScoreResponse(
        portrait=result["portrait"],
        portrait_label=result["portrait_label"],
        model_source=result["model_source"],
        interaction_interest_score=result["interaction_interest_score"],
        top_products=[PropensityProductItem(**item) for item in result["top_products"]],
        all_products=[PropensityProductItem(**item) for item in result["all_products"]],
    )
