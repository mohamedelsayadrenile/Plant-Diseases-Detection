from PIL import Image
from ultralytics import YOLO

from src.core.config import settings
from src.core.logging import get_logger
from src.services.disease_detection.interface import DetectionResult
from src.services.disease_detection.messages import known_classes, message_for

logger = get_logger(__name__)

# YOLO's own conf floor. Kept far below YOLO_DISEASE_CONF so sub-threshold
# detections stay visible to the caller and can be escalated to the fallback
# provider; below this they are noise. Not a tuning knob -- YOLO_DISEASE_CONF is
# the single threshold that decides what we trust.
_RAW_CONF_FLOOR = 0.05

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

class YoloProvider:
    """The local model. Primary provider -- always asked first."""

    name = "yolo"

    def __init__(self) -> None:
        self._model = YOLO(settings.DISEASE_MODEL_PATH)
        self._model.to(settings.DEVICE)
        logger.info("disease model loaded from %s", settings.DISEASE_MODEL_PATH)
        self._check_message_coverage()

    def _check_message_coverage(self) -> None:
        """Warn if the advice sheet and the checkpoint have drifted apart.

        A class the sheet does not cover would silently serve a null message,
        which is indistinguishable from a healthy verdict at the API. Checking
        once at startup turns that into a log line instead of a quiet gap.
        """
        model_names = set(self._model.names.values())
        sheet_names = known_classes()
        if missing := sorted(model_names - sheet_names):
            logger.warning("no arabic message for %d class(es): %s", len(missing), missing)
        if stale := sorted(sheet_names - model_names):
            logger.warning("arabic sheet has %d unknown class(es): %s", len(stale), stale)

    async def detect(
        self, image: Image.Image, raw: bytes, content_type: str
    ) -> DetectionResult | None:
        """Highest-confidence detection, or None if nothing was detected.

        Detections below YOLO_DISEASE_CONF are returned with their confidence
        so the caller can escalate; the caller decides what to trust.
        """
        result = self._model.predict(
            source=image,
            imgsz=settings.IMAGE_SIZE,
            conf=min(_RAW_CONF_FLOOR, settings.YOLO_DISEASE_CONF),
            device=settings.DEVICE,
            save=False,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            logger.info("yolo: no detections above %.2f", _RAW_CONF_FLOOR)
            return None

        best = int(result.boxes.conf.argmax())
        class_id = int(result.boxes.cls[best])
        name = result.names[class_id]
        confidence = float(result.boxes.conf[best])
        logger.info("yolo: %s (%.2f)", name, confidence)

        if class_id in HEALTHY_CLASS_IDS:
            return DetectionResult(None, True, confidence, self.name)
        return DetectionResult(name, False, confidence, self.name, message=message_for(name))
