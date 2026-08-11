from PIL import Image

from src.core.config import settings
from src.core.logging import get_logger
from src.services.disease_detection.interface import DetectionResult
from src.services.disease_detection.providers.kindwise import KindwiseProvider
from src.services.disease_detection.providers.yolo import YoloProvider

logger = get_logger(__name__)

_yolo: YoloProvider | None = None
_kindwise: KindwiseProvider | None = None


def startup() -> None:
    """Build the providers once, at app startup."""
    global _yolo, _kindwise
    _yolo = YoloProvider()
    if settings.KINDWISE_API_KEY:
        _kindwise = KindwiseProvider()
        logger.info("kindwise fallback enabled (%s)", settings.KINDWISE_API_URL)
    else:
        # An optional credential must not stop the app booting: the local model
        # is the primary provider and works on its own.
        logger.warning("KINDWISE_API_KEY not set - kindwise fallback disabled")


async def shutdown() -> None:
    global _kindwise
    if _kindwise is not None:
        await _kindwise.aclose()
        _kindwise = None


def fallback_enabled() -> bool:
    return _kindwise is not None


async def detect_disease(
    image: Image.Image, raw: bytes, content_type: str
) -> DetectionResult | None:
    """Local model first; Kindwise only when it is below YOLO_DISEASE_CONF.

    Returns None when no provider produced an answer we are willing to stand
    behind.
    """
    local = await _yolo.detect(image, raw, content_type)

    # A confident local answer -- disease or healthy -- always wins.
    if local is not None and local.confidence >= settings.YOLO_DISEASE_CONF:
        return local

    if _kindwise is None:
        # Sub-threshold answers are never served, so behaviour with the
        # fallback disabled matches the pre-change contract.
        return None

    logger.info(
        "escalating to kindwise (yolo=%s)",
        f"{local.disease or 'healthy'} {local.confidence:.2f}" if local else "no detection",
    )
    # The local answer was rejected on confidence grounds, and a failed call
    # does not make it trustworthy, so it is never served as a consolation.
    return await _kindwise.detect(image, raw, content_type)
