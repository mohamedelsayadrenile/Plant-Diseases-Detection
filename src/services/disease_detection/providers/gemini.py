import httpx
from google import genai
from google.genai import errors, types
from PIL import Image
from pydantic import BaseModel

from src.core.config import settings
from src.core.logging import get_logger
from src.services.disease_detection.interface import DetectionResult

logger = get_logger(__name__)


# NOTE: this model is converted into the schema sent to Gemini, so its
# docstring reaches the model as the schema description -- keep it addressed to
# the model. Do not set `extra`: that emits `additionalProperties`, which the
# Gemini Developer API rejects.
class GeminiAnswer(BaseModel):
    """A plant disease diagnosis for a single photograph."""

    is_plant: bool
    is_healthy: bool
    disease: str | None = None
    message: str | None = None


# The `message` wording mirrors the local advice sheet (see messages.py) so an
# answer reads the same to the grower whichever provider produced it.
_ADVICE_SPEC = (
    "Write it in Modern Standard Arabic in two parts: first "
    "'السبب المحتمل:' naming the pathogen and the conditions that favour it, "
    "then a newline and 'العلاج الموصى به:' giving concrete steps -- cultural "
    "practices and named active ingredients. Two to four sentences in total. "
    "Latin pathogen and active-ingredient names stay in Latin script. Use plain "
    "text only -- no markdown, no asterisks, no bold or italic markers."
)

_PROMPT = (
    "You are a plant pathologist examining a single photograph.\n"
    "- is_plant: true only if the image shows a real plant or part of one.\n"
    "- is_healthy: true if the plant looks healthy, false if it shows disease "
    "symptoms. Ignored when is_plant is false.\n"
    "- disease: the common name of the single most likely disease, or null if "
    "the plant is healthy, it is not a plant, or you cannot tell.\n"
    f"- message: advice for the grower about that disease. {_ADVICE_SPEC} "
    "Null whenever disease is null.\n"
    "Answer only with the JSON object."
)

# Text-only prompt for a disease another provider already named. The name is
# trusted rather than re-diagnosed: the caller accepted that verdict on its own
# confidence, and this call exists only to dress it in Arabic advice.
_DESCRIBE_PROMPT = (
    "You are a plant pathologist. A grower's plant has been diagnosed with: "
    "{disease}.\n"
    f"Write advice for the grower about this disease. {_ADVICE_SPEC}\n"
    "Answer with the Arabic text only -- no JSON, no preamble, no markdown."
)


class GeminiProvider:
    """Last-resort provider, asked only when YOLO and Kindwise are both unsure."""

    name = "gemini"

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiAnswer,
            http_options=types.HttpOptions(timeout=settings.GEMINI_TIMEOUT_MS),
        )
        # `describe` wants prose, not JSON, so it needs a config without the
        # schema. Built here rather than per call.
        self._text_config = types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=settings.GEMINI_TIMEOUT_MS),
        )

    async def aclose(self) -> None:
        await self._client.aio.aclose()

    async def _generate(
        self, contents: list, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse | None:
        """One Gemini call, with every failure turned into None.

        Shared by `detect` and `describe` so both fail the same way: this
        provider never raises, and a failed message must not cost us a verdict.
        """
        try:
            return await self._client.aio.models.generate_content(
                model=settings.GEMINI_MODEL, contents=contents, config=config
            )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            # The SDK does not wrap transport errors -- they arrive raw from
            # httpx, so they are caught separately from APIError below. The type
            # name is logged because these often carry an empty message.
            logger.warning("gemini transport failed: %s %s", type(exc).__name__, exc)
        except errors.APIError as exc:
            logger.warning("gemini API error %s: %s", exc.code, exc.message)
        except Exception:
            logger.exception("gemini call failed unexpectedly")
        return None

    async def describe(self, disease: str) -> str | None:
        """Write the Arabic advice for a disease another provider named.

        Kindwise and Gemini answer from open label spaces, so their diseases
        cannot be looked up in the local sheet the way the local model's fixed
        classes are. Returns None on any failure -- the caller keeps its verdict
        and serves it without a message.
        """
        response = await self._generate(
            [_DESCRIBE_PROMPT.format(disease=disease)], self._text_config
        )
        if response is None:
            return None

        # `.text` is None when the reply was blocked or came back empty.
        message = (response.text or "").strip()
        if not message:
            logger.warning("gemini returned no message for %s", disease)
            return None
        return message

    async def detect(
        self, image: Image.Image, raw: bytes, content_type: str
    ) -> DetectionResult | None:
        """Ask Gemini to name the disease.

        Returns None on any failure or non-answer. Like every provider, this
        never raises.
        """
        response = await self._generate(
            [types.Part.from_bytes(data=raw, mime_type=content_type), _PROMPT],
            self._config,
        )
        if response is None:
            return None

        # `parsed` is silently None -- no exception, no log -- when the reply is
        # blocked, empty, or fails schema validation, so it must be checked.
        answer = response.parsed
        if answer is None:
            logger.warning("gemini returned no parsable answer")
            return None

        logger.info(
            "gemini: is_plant=%s is_healthy=%s disease=%s",
            answer.is_plant,
            answer.is_healthy,
            answer.disease,
        )

        # confidence stays None throughout: a self-reported score from an LLM is
        # not comparable to the calibrated ones the other two providers return.
        if not answer.is_plant:
            return DetectionResult(None, None, None, self.name, is_plant=False)
        if answer.is_healthy:
            return DetectionResult(None, True, None, self.name)
        if not answer.disease:
            return None  # diseased but unnamed is not an answer
        # The message rides along only here. The model is told to null it for
        # the other outcomes, but those branches drop it regardless: advice
        # without a named disease is not something we want to serve.
        return DetectionResult(
            answer.disease, False, None, self.name, message=answer.message
        )
