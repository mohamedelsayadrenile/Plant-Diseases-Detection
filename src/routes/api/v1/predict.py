import io
from dataclasses import asdict
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

# Derived from the validated extension rather than the client's Content-Type
# header, which is browser-supplied and often wrong or absent.
MIME_BY_EXTENSION = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Kept as uploaded: the fallback provider is sent these bytes untouched,
    # since re-encoding is known to change the answer (see README).
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")

    if not has_leaf(image):
        logger.info("no leaf detected in %s", file.filename)
        return PredictionResponse(disease=None, is_healthy=None)

    result = await detect_disease(image, raw, MIME_BY_EXTENSION[extension])
    if result is None:
        return PredictionResponse(disease=None, is_healthy=None)
    return PredictionResponse(**asdict(result))
