from pydantic import BaseModel


class PredictionResponse(BaseModel):
    # The outermost gate: false when the image is judged not to contain a
    # plant, in which case the remaining fields are all null.
    is_plant: bool
    disease: str | None
    is_healthy: bool | None
    confidence: float | None = None
    # Which provider produced the answer: "yolo", "kindwise", or null when
    # neither did. The shape is the same whichever one answered.
    source: str | None = None
    # Arabic advice for the grower -- the likely cause of the disease and the
    # recommended treatment. Null unless `disease` is set: a healthy plant, a
    # non-plant image, and an unresolved one all have no advice to give.
    message: str | None = None
