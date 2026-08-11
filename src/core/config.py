from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    LEAF_MODEL_PATH: str = str(ROOT / "yolo11x_leaf.pt")
    DISEASE_MODEL_PATH: str = str(ROOT / "PlantDiseaseDetection.pt")
    LEAF_CONF: float = 0.15
    # Acceptance threshold applied in Python, not YOLO's own conf floor. Below
    # this the local answer is not trusted and Kindwise is asked instead.
    YOLO_DISEASE_CONF: float = 0.35
    DEVICE: str = "cuda:0"
    IMAGE_SIZE: int = 640
    KINDWISE_API_URL: str = "https://crop.kindwise.com/api/v1"
    KINDWISE_API_KEY: str = ""
    KINDWISE_TIMEOUT: float = 20.0


settings = Settings()
