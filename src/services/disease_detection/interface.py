from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """What a provider found. `source` names the provider that produced it."""

    disease: str | None  # None with is_healthy=True means a healthy leaf
    is_healthy: bool | None
    confidence: float | None
    source: str
    # Whether the provider considers the image to contain a plant at all.
    # Defaults to True because a provider that recognised a leaf class has
    # implicitly answered it; only Kindwise reports on this independently, and
    # only it can overturn the leaf gate by returning False.
    is_plant: bool = True


class DiseaseProvider(Protocol):
    name: str

    async def detect(
        self, image: Image.Image, raw: bytes, content_type: str
    ) -> DetectionResult | None:
        """Detect a disease, or None if this provider has no usable answer.

        The arguments are the union of what the providers need: the local model
        reads `image`, Kindwise reads `raw` and `content_type`. Kindwise must be
        given the bytes exactly as uploaded -- re-encoding is known to change
        the answer (see README).
        """
