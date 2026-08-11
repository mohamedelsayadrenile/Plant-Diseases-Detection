from pydantic import BaseModel


class PredictionResponse(BaseModel):
    disease: str | None
    is_healthy: bool | None
    confidence: float | None = None
    # Which provider produced the answer: "yolo", "kindwise", or null when
    # neither did. The shape is the same whichever one answered.
    source: str | None = None
