import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

LEAF_MODEL_PATH = os.getenv("LEAF_MODEL_PATH", str(ROOT / "yolo11x_leaf.pt"))
DISEASE_MODEL_PATH = os.getenv("DISEASE_MODEL_PATH", str(ROOT / "PlantDiseaseDetection.pt"))

LEAF_CONF = float(os.getenv("LEAF_CONF", 0.15))
DISEASE_CONF = float(os.getenv("DISEASE_CONF", 0.25))

DEVICE = os.getenv("DEVICE", "cuda:0")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 640))
