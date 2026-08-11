from PIL import Image
from ultralytics import YOLO

from src.core import config
from src.core.logging import get_logger

logger = get_logger(__name__)

_model: YOLO | None = None


def load_model() -> None:
    global _model
    _model = YOLO(config.LEAF_MODEL_PATH)
    _model.to(config.DEVICE)
    logger.info("leaf model loaded from %s", config.LEAF_MODEL_PATH)


def has_leaf(image: Image.Image) -> bool:
    """True if the image contains at least one leaf above LEAF_CONF."""
    result = _model.predict(
        source=image,
        imgsz=config.IMAGE_SIZE,
        conf=config.LEAF_CONF,
        device=config.DEVICE,
        save=False,
        verbose=False,
    )[0]
    count = 0 if result.boxes is None else len(result.boxes)
    logger.info("leaf detection: %d leaf(s)", count)
    return count > 0
