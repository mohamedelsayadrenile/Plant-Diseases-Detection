from PIL import Image
from ultralytics import YOLO

from src.core import config
from src.core.logging import get_logger

logger = get_logger(__name__)

# Class IDs whose label means a healthy leaf rather than a disease. Listed
# explicitly because the names cannot be classified by string rules: several
# disease classes also end in " leaf" (109 "Corn rust leaf", 112 "Tomato blight
# leaf") or contain it (4 "corn northern leaf blight", 42 "tomato leaf mold").
HEALTHY_CLASS_IDS = frozenset(
    {
        # "<crop> leaf" classes
        1, 2, 3, 5, 7, 9, 14, 17, 19, 20, 21, 23, 24, 26, 30, 33, 38, 54,
        59, 65, 68, 70, 72, 73, 74, 75, 77, 78, 79, 84,
        # explicitly named healthy classes
        91,   # Cassava Healthy
        98,   # Corn Healthy
        113,  # Tomato healthy
    }
)

_model: YOLO | None = None


def load_model() -> None:
    global _model
    _model = YOLO(config.DISEASE_MODEL_PATH)
    _model.to(config.DEVICE)
    logger.info("disease model loaded from %s", config.DISEASE_MODEL_PATH)


def detect_disease(image: Image.Image) -> tuple[str | None, bool | None]:
    """Return (disease name, is_healthy) for the highest-confidence detection.

    (None, True)  -> a healthy-leaf class was detected
    (None, None)  -> nothing detected above DISEASE_CONF
    """
    result = _model.predict(
        source=image,
        imgsz=config.IMAGE_SIZE,
        conf=config.DISEASE_CONF,
        device=config.DEVICE,
        save=False,
        verbose=False,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        logger.info("disease detection: no detections above %.2f", config.DISEASE_CONF)
        return None, None

    best = int(result.boxes.conf.argmax())
    class_id = int(result.boxes.cls[best])
    name = result.names[class_id]
    confidence = float(result.boxes.conf[best])
    logger.info("disease detection: %s (%.2f)", name, confidence)

    if class_id in HEALTHY_CLASS_IDS:
        return None, True
    return name, False
