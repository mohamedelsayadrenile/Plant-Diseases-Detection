import io
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from src.core.logging import get_logger
from src.models.schema.prediction import PredictionResponse
from src.services.disease_detection import detect_disease
from src.services.leaf_detection import has_leaf

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["prediction"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")

    if not has_leaf(image):
        logger.info("no leaf detected in %s", file.filename)
        return PredictionResponse(disease=None, is_healthy=None)

    disease, is_healthy = detect_disease(image)
    return PredictionResponse(disease=disease, is_healthy=is_healthy)
