from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.config import settings
from src.core.logging import get_logger
from src.routes.api.v1 import predict
from src.services import disease_detection, leaf_detection

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading models on %s", settings.DEVICE)
    leaf_detection.load_model()
    disease_detection.startup()
    logger.info("models loaded")
    try:
        yield
    finally:
        await disease_detection.shutdown()


app = FastAPI(
    title="Plant Disease Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(predict.router)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "device": settings.DEVICE,
        "fallback": "kindwise" if disease_detection.fallback_enabled() else "disabled",
    }
