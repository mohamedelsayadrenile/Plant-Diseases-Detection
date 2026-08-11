from PIL import Image
from ultralytics import YOLO

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

_model: YOLO | None = None


def load_model() -> None:
    global _model
    _model = YOLO(settings.LEAF_MODEL_PATH)
    _model.to(settings.DEVICE)
    logger.info("leaf model loaded from %s", settings.LEAF_MODEL_PATH)


def has_leaf(image: Image.Image) -> bool:
    """True if the image contains at least one leaf above LEAF_CONF."""
    result = _model.predict(
        source=image,
        imgsz=settings.IMAGE_SIZE,
        conf=settings.LEAF_CONF,
        device=settings.DEVICE,
        save=False,
        verbose=False,
    )[0]
    count = 0 if result.boxes is None else len(result.boxes)
    confidences = sorted((round(float(c), 2) for c in result.boxes.conf), reverse=True) if count else []
    logger.info("leaf detection: %d leaf(s) %s", count, confidences)
    return count > 0
