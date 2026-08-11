from PIL import Image

from src.core.config import settings
from src.core.logging import get_logger
from src.services.disease_detection.interface import DetectionResult
from src.services.disease_detection.providers.gemini import GeminiProvider
from src.services.disease_detection.providers.kindwise import KindwiseProvider
from src.services.disease_detection.providers.yolo import YoloProvider

logger = get_logger(__name__)

_yolo: YoloProvider | None = None
_kindwise: KindwiseProvider | None = None
_gemini: GeminiProvider | None = None


def startup() -> None:
    """Build the providers once, at app startup.

    Only the local model is required. Each remote provider is built when its
    credential is present and skipped otherwise, so a missing optional key
    degrades the chain instead of stopping the app booting.
    """
    global _yolo, _kindwise, _gemini
    _yolo = YoloProvider()

    if settings.KINDWISE_API_KEY:
        _kindwise = KindwiseProvider()
        logger.info("kindwise fallback enabled (%s)", settings.KINDWISE_API_URL)
    else:
        logger.warning("KINDWISE_API_KEY not set - kindwise fallback disabled")

    if settings.GEMINI_API_KEY:
        _gemini = GeminiProvider()
        logger.info("gemini fallback enabled (%s)", settings.GEMINI_MODEL)
    else:
        logger.warning("GEMINI_API_KEY not set - gemini fallback disabled")


async def shutdown() -> None:
    global _kindwise, _gemini
    if _kindwise is not None:
        await _kindwise.aclose()
        _kindwise = None
    if _gemini is not None:
        await _gemini.aclose()
        _gemini = None


def enabled_providers() -> list[str]:
    """The providers that will be tried, in chain order."""
    return [p.name for p in (_yolo, _kindwise, _gemini) if p is not None]


def _confident(result: DetectionResult | None, threshold: float) -> bool:
    """True when a provider produced an answer we trust at `threshold`."""
    return (
        result is not None
        and result.confidence is not None
        and result.confidence >= threshold
    )


def _describe(result: DetectionResult | None) -> str:
    if result is None:
        return "no answer"
    label = result.disease or ("healthy" if result.is_healthy else "none")
    return f"{label} {result.confidence:.2f}" if result.confidence is not None else label


async def detect_disease(
    image: Image.Image, raw: bytes, content_type: str
) -> DetectionResult | None:
    """Ask each provider in turn, stopping at the first answer we trust.

    Returns None when no provider produced an answer we are willing to stand
    behind. An answer rejected on confidence grounds is never served, so a
    failed escalation returns None rather than the answer that triggered it.
    """
    result = await _yolo.detect(image, raw, content_type)
    if _confident(result, settings.YOLO_DISEASE_CONF):
        return result
    previous = "yolo"

    if _kindwise is not None:
        logger.info("escalating to kindwise (%s=%s)", previous, _describe(result))
        result = await _kindwise.detect(image, raw, content_type)
        # "not a plant" is a definitive verdict rather than a weak one, so it
        # ends the chain instead of buying a third opinion on a photo of a bicycle.
        if result is not None and not result.is_plant:
            return result
        if _confident(result, settings.KINDWISE_CONF):
            return result
        previous = "kindwise"

    if _gemini is not None:
        logger.info("escalating to gemini (%s=%s)", previous, _describe(result))
        return await _gemini.detect(image, raw, content_type)

    return None
