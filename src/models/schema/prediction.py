from pydantic import BaseModel


class PredictionResponse(BaseModel):
    disease: str | None
    is_healthy: bool | None
