"""FastAPI: демо ступени 1 — классификация бизнес-портрета."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.schemas import (
    ConfigResponse,
    GenerateSalesArgumentRequest,
    GenerateMetricsRequest,
    GenerateStage2SalesArgumentRequest,
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
    RenderPropensityFeaturePromptRequest,
    RenderedPromptResponse,
    RenderMetricsPromptRequest,
    RenderSalesArgPromptRequest,
    RenderStage2SalesArgPromptRequest,
    SalesArgumentResponse,
    SalesArgumentItem,
    SalesArgumentsConfig,
    ShapFeatureItem,
    Stage2SalesArgumentResponse,
)
from config.sales_arguments import INTERACTION_TYPES, MOCK_SALES_ARGUMENTS
from config.propensity import PROPENSITY_FEATURE_LABELS
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
from services.propensity_feature_generator import (
    generate_propensity_features,
    render_propensity_feature_prompt,
)
from services.propensity_scorer import score_propensity
from services.random_metrics_generator import generate_metrics_random
from services.sales_argument_generator import (
    generate_sales_argument,
    generate_stage2_argument,
)
from services.sales_arg_renderer import (
    render_sales_arg_prompt,
    render_stage2_sales_arg_prompt,
)
from dotenv import load_dotenv

load_dotenv()

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
        feature_labels={**PROPENSITY_FEATURE_LABELS, **FEATURE_LABELS},
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
        top5_feature_importance=[
            ShapFeatureItem(**item) for item in result["top5_feature_importance"]
        ],
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
        raise HTTPException(
            status_code=500, detail=f"Ошибка генерации sales-аргумента: {e}"
        )

    return SalesArgumentResponse(**result)


@app.post("/api/v1/metrics/render-prompt", response_model=RenderedPromptResponse)
async def render_metrics_prompt_endpoint(body: RenderMetricsPromptRequest):
    """Рендерить промпт для генерации метрик без вызова Mistral (для превью в UI)."""
    if body.channel not in ("digital", "voice"):
        raise HTTPException(
            status_code=422, detail="channel должен быть 'digital' или 'voice'"
        )
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
        raise HTTPException(
            status_code=422, detail="channel должен быть 'digital' или 'voice'"
        )
    if body.method not in ("llm", "random"):
        raise HTTPException(
            status_code=422, detail="method должен быть 'llm' или 'random'"
        )
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


@app.post("/api/v1/propensity/render-feature-prompt", response_model=RenderedPromptResponse)
async def render_propensity_feature_prompt_endpoint(body: RenderPropensityFeaturePromptRequest):
    """Рендерить промпт генерации признаков для Stage 2 без вызова Mistral."""
    try:
        prompt = render_propensity_feature_prompt(
            classification=body.classification,
            client_features=body.client_features,
            metrics_result=body.metrics_result,
            sales_argument=body.sales_argument,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка рендеринга промпта признаков: {e}")
    return RenderedPromptResponse(rendered_prompt=prompt)


@app.post("/api/v1/propensity/score", response_model=PropensityScoreResponse)
async def score_propensity_endpoint(body: PropensityScoreRequest):
    """Сгенерировать признаки клиента и оценить склонность к продуктам."""
    try:
        feature_generation = generate_propensity_features(
            classification=body.classification,
            client_features=body.client_features,
            metrics_result=body.metrics_result,
            sales_argument=body.sales_argument,
        )
        result = score_propensity(
            classification=body.classification,
            client_features=body.client_features,
            metrics_result=body.metrics_result,
            top_k=body.top_k,
            generated_features=feature_generation["features"],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка скоринга склонности: {e}")

    return PropensityScoreResponse(
        portrait=result["portrait"],
        portrait_label=result["portrait_label"],
        model_source=result["model_source"],
        feature_source=result["feature_source"],
        generated_features=result["scoring_features"],
        feature_generation_reasoning=feature_generation["reasoning_summary"],
        feature_generation_prompt=feature_generation["rendered_prompt"],
        feature_generation_system_prompt=feature_generation["system_prompt"],
        feature_generation_raw_response=feature_generation["raw_llm_response"],
        interaction_interest_score=result["interaction_interest_score"],
        top_products=[PropensityProductItem(**item) for item in result["top_products"]],
        all_products=[PropensityProductItem(**item) for item in result["all_products"]],
    )


@app.post(
    "/api/v1/sales-args/render-prompt-stage2", response_model=RenderedPromptResponse
)
async def render_stage2_sales_arg_prompt_endpoint(
    body: RenderStage2SalesArgPromptRequest,
):
    """Рендерить шаблон Stage 2 sales-аргумента без вызова Mistral (для превью в UI)."""
    try:
        prompt = render_stage2_sales_arg_prompt(
            classification=body.classification,
            interaction_type=body.interaction_type,
            client_features=body.client_features,
            propensity_product=body.propensity_product,
            stage1_argument=body.stage1_argument,
            stage1_metrics=body.stage1_metrics,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка рендеринга промпта Stage 2: {e}"
        )
    return RenderedPromptResponse(rendered_prompt=prompt)


@app.post(
    "/api/v1/sales-args/generate-stage2", response_model=Stage2SalesArgumentResponse
)
async def generate_stage2_sales_arg_endpoint(body: GenerateStage2SalesArgumentRequest):
    """Сгенерировать персонализированный Stage 2 sales-аргумент через Mistral."""
    try:
        result = generate_stage2_argument(
            classification=body.classification,
            interaction_type=body.interaction_type,
            client_features=body.client_features,
            propensity_product=body.propensity_product,
            stage1_argument=body.stage1_argument,
            stage1_metrics=body.stage1_metrics,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка генерации Stage 2 аргумента: {e}"
        )

    return Stage2SalesArgumentResponse(**result)
