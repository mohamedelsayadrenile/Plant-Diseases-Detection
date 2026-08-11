import httpx
from PIL import Image

from src.core.config import settings
from src.core.logging import get_logger
from src.services.disease_detection.interface import DetectionResult

logger = get_logger(__name__)

_ENDPOINT = "/identification"
_LANGUAGE = "en"

# Kindwise reports a healthy crop as an ordinary disease suggestion rather than
# a flag, so healthy names have to be recognised the same way the local model's
# HEALTHY_CLASS_IDS are. Matched lowercased and stripped. Unverified against a
# real healthy-leaf response -- until it is, an unlisted name simply reads as a
# disease, never as a false "healthy".
HEALTHY_SUGGESTION_NAMES = frozenset({"healthy", "no disease", "healthy plant"})


class KindwiseProvider:
    """Fallback provider. Asked only when the local model is not confident."""

    name = "kindwise"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.KINDWISE_API_URL,
            headers={"Api-Key": settings.KINDWISE_API_KEY},
            timeout=settings.KINDWISE_TIMEOUT,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def detect(
        self, image: Image.Image, raw: bytes, content_type: str
    ) -> DetectionResult | None:
        """Second opinion from the Kindwise API.

        Returns None on any failure or non-answer. This never raises: a fallback
        provider must not be able to break the endpoint.
        """
        try:
            response = await self._client.post(
                _ENDPOINT,
                # language goes in the query string -- sent as a form field the
                # API reads it as an image modifier and rejects it with 400.
                params={"language": _LANGUAGE},
                files={"images": ("upload", raw, content_type)},
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException:
            logger.warning("kindwise timed out after %.1fs", settings.KINDWISE_TIMEOUT)
            return None
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "kindwise HTTP %s: %s", exc.response.status_code, exc.response.text[:500]
            )
            return None
        except httpx.RequestError as exc:
            logger.warning("kindwise request failed: %s", exc)
            return None

        # Kept separate from the transport block so a parsing bug is never
        # reported as a network problem.
        try:
            return self._to_result(payload)
        except Exception:
            logger.exception("kindwise response could not be parsed")
            return None

    def _to_result(self, payload: dict) -> DetectionResult | None:
        result = payload.get("result") or {}

        if (result.get("is_plant") or {}).get("binary") is False:
            logger.info("kindwise: not a plant")
            return None

        suggestions = (result.get("disease") or {}).get("suggestions") or []
        if not suggestions:
            logger.info("kindwise: no disease suggestions")
            return None

        top = suggestions[0]  # the API returns them sorted by probability desc
        name = str(top.get("name", "")).strip()
        probability = float(top.get("probability", 0.0))
        logger.info("kindwise: %s (%.3f)", name, probability)

        if name.lower() in HEALTHY_SUGGESTION_NAMES:
            return DetectionResult(None, True, probability, self.name)
        return DetectionResult(name, False, probability, self.name)
