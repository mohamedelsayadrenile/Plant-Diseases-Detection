# Plant Disease Detection API

A small FastAPI service that runs two YOLO models sequentially over an uploaded plant image:

```
Image
  ↓
Leaf YOLO (yolo11x_leaf.pt)  ── no leaf ──→  {"disease": null, "is_healthy": null}
  ↓ leaf detected
Disease YOLO (PlantDiseaseDetection.pt)
  ↓
{"disease": "...", "is_healthy": false}
```

The leaf model is purely a gate. If it finds no leaf, the disease model never runs. If it passes, the disease model receives the **original uploaded image** — not leaf crops.

The API returns JSON only. It never returns an annotated or processed image.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- The two model checkpoints in the project root (or wherever `.env` points):
  - `yolo11x_leaf.pt` (~109 MB) — 1 class: `leaf`
  - `PlantDiseaseDetection.pt` (~436 MB) — 116 classes
- A CUDA GPU is optional; set `DEVICE=cpu` to run on CPU.

## Setup

```bash
uv sync
cp .env.example .env      # then edit the model paths
```

## Configuration

All configuration lives in `.env` and is loaded by `src/core/config.py`. Nothing is configurable per request.

| Variable | Default | Description |
|---|---|---|
| `LEAF_MODEL_PATH` | `./yolo11x_leaf.pt` | Path to the leaf detection checkpoint |
| `DISEASE_MODEL_PATH` | `./PlantDiseaseDetection.pt` | Path to the disease detection checkpoint |
| `LEAF_CONF` | `0.15` | Confidence threshold for the leaf gate |
| `DISEASE_CONF` | `0.25` | Confidence threshold for disease detection |
| `DEVICE` | `cuda:0` | Inference device — `cuda:0`, `cuda:1`, or `cpu` |
| `IMAGE_SIZE` | `640` | Inference image size (both models were trained at 640) |

## Running

```bash
uv run uvicorn src.main:app --reload
```

Both models load once at startup, not per request. With `--reload` every code change re-loads ~545 MB of checkpoints, so expect a pause after each save.

## Endpoints

### `POST /api/v1/predict`

Multipart upload. The request contains only the image file.

| Field | Type | Description |
|---|---|---|
| `file` | file | `.jpg`, `.jpeg`, `.png`, or `.webp` |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "file=@potato_late.jpeg"
```

**Response** — always two fields:

```json
{ "disease": "potato early blight", "is_healthy": false }
```

| Case | `disease` | `is_healthy` |
|---|---|---|
| Disease detected | class name | `false` |
| Healthy leaf class detected | `null` | `true` |
| No leaf detected | `null` | `null` |
| Leaf found, nothing above `DISEASE_CONF` | `null` | `null` |

When the disease model returns several detections, the highest-confidence one is reported.

**Errors** (`400`):

```json
{ "detail": "Unsupported file type. Allowed: ['.jpeg', '.jpg', '.png', '.webp']" }
{ "detail": "Invalid or corrupt image file." }
```

### `GET /health`

```json
{ "status": "ok", "device": "cuda:0" }
```

## Healthy vs. disease classes

The disease model has 116 classes, mixing actual diseases with healthy-leaf labels (`tomato leaf`, `apple leaf`, `Corn Healthy`, …). `src/services/disease_detection.py` holds an explicit set of the 33 healthy class IDs.

The set is explicit rather than rule-based on purpose. A rule like `name.endswith(" leaf")` would misclassify `Corn rust leaf` (109) and `Tomato blight leaf` (112) as healthy, and a "contains leaf" rule would break another 25 disease labels such as `corn northern leaf blight` and `tomato leaf mold`. Reporting a blight as healthy is this service's worst failure mode, so the IDs are listed one by one.

Three classes name no pathogen but are not healthy either — `Corn Insects Damages` (99), `Corn Purple Discoloration` (101), `Corn Yellowing` (107). They are deliberately excluded from the healthy set, so they report as the class name with `is_healthy: false`.

## Project structure

```
src/
├── main.py                       # FastAPI app, model loading at startup, GET /health
├── core/
│   ├── config.py                 # .env configuration
│   └── logging.py                # logger setup
├── models/schema/
│   └── prediction.py             # PredictionResponse
├── services/
│   ├── leaf_detection.py         # load_model(), has_leaf()
│   └── disease_detection.py      # load_model(), detect_disease(), HEALTHY_CLASS_IDS
└── routes/api/v1/
    └── predict.py                # POST /api/v1/predict
```

## Known limitations

- **The leaf gate produces false positives.** `yolo11x_leaf.pt` detects a "leaf" in a solid blue image at 0.883 confidence. Raising `LEAF_CONF` does not help at that confidence — it is a limitation of the checkpoint, not the threshold.
- **`{"disease": null, "is_healthy": null}` is ambiguous.** It covers both "no leaf detected" and "leaf found but no detection above `DISEASE_CONF`". The application log distinguishes the two.
- **Re-encoding can change the result.** The same photo saved as `.webp` returned `potato late blight` (0.68) where the JPEG returned `potato early blight` (0.64) — lossy compression is enough to reorder two closely scored classes.
- The 116-class list contains near-duplicates from two merged training sets (`corn rust` / `Corn rust leaf`, `corn smut` / `Corn Smut`), so the reported class name may vary in capitalization and wording between similar images.
