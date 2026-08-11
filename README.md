# Plant Disease Detection API

A small FastAPI service that runs two YOLO models sequentially over an uploaded plant image, with a remote API as a fallback second opinion:

```
Image
  ↓
Leaf YOLO (yolo11x_leaf.pt)  ── no leaf ──→  {"is_plant": false, ...nulls}
  ↓ leaf detected
Disease YOLO (PlantDiseaseDetection.pt)     ── conf ≥ YOLO_DISEASE_CONF ──→  {"source": "yolo", ...}
  ↓ below YOLO_DISEASE_CONF, or nothing detected
Kindwise API                 ── not a plant ──→  {"is_plant": false, ...nulls}
  ↓ answered ──→ {"source": "kindwise", ...}
  ↓ unavailable, or also unsure
{"is_plant": true, "disease": null, "is_healthy": null}
```

The leaf model is purely a gate. If it finds no leaf, the disease model never runs. If it passes, the disease model receives the **original uploaded image** — not leaf crops.

The local model is always the primary provider and is always asked first. Kindwise is called only when the local model's best detection is below `YOLO_DISEASE_CONF`, or when it detected nothing at all — so a confident local answer never costs an API call. If Kindwise is unreachable, times out, errors, or has no API key configured, the endpoint logs a warning and returns nulls; it never fails the request, and it never falls back to the low-confidence local answer that triggered the escalation in the first place.

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
| `YOLO_DISEASE_CONF` | `0.35` | Local detections at or above this are trusted; below it the Kindwise fallback is asked instead |
| `DEVICE` | `cuda:0` | Inference device — `cuda:0`, `cuda:1`, or `cpu` |
| `IMAGE_SIZE` | `640` | Inference image size (both models were trained at 640) |
| `KINDWISE_API_URL` | `https://crop.kindwise.com/api/v1` | Base URL of the fallback provider |
| `KINDWISE_API_KEY` | *(empty)* | Fallback API key. Leave empty to disable the fallback entirely |
| `KINDWISE_TIMEOUT` | `20.0` | Per-request timeout, in seconds, for the fallback call |

`YOLO_DISEASE_CONF` is an **acceptance** threshold applied in Python, not the `conf=` floor handed to
YOLO. The model itself runs at a fixed low floor (`_RAW_CONF_FLOOR = 0.05` in
`src/services/disease_detection/providers/yolo.py`) so that weak detections stay visible and can be
escalated; without that, anything below the threshold would be discarded inside the model and be
indistinguishable from "nothing found". The low floor cannot change which detection wins — NMS keeps
the highest-scoring box of a cluster either way.

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

**Response** — always the same five fields, whichever provider answered:

```json
{
  "is_plant": true,
  "disease": "potato early blight",
  "is_healthy": false,
  "confidence": 0.636,
  "source": "yolo"
}
```

| Case | `is_plant` | `disease` | `is_healthy` | `confidence` | `source` |
|---|---|---|---|---|---|
| Local model confident | `true` | class name | `false` | its score | `"yolo"` |
| Local model confident, healthy class | `true` | `null` | `true` | its score | `"yolo"` |
| Escalated, Kindwise answered | `true` | disease name | `false` | its probability | `"kindwise"` |
| Escalated, Kindwise says not a plant | `false` | `null` | `null` | `null` | `"kindwise"` |
| No leaf detected | `false` | `null` | `null` | `null` | `null` |
| Escalated, Kindwise unavailable or unsure | `true` | `null` | `null` | `null` | `null` |
| Local model unsure, fallback disabled | `true` | `null` | `null` | `null` | `null` |

`is_plant` is the outermost gate: when it is `false`, every other field is `null`. It is `false` in
exactly two situations — the leaf model found no leaf, or Kindwise's own plant check overruled the
leaf model. Otherwise it is `true`, meaning the leaf gate passed and nothing contradicted it. Note
that a `true` backed only by the leaf gate is weak evidence; see the limitations below.

`source` names the provider that produced the answer and `confidence` is that provider's own score,
so the two are only comparable within a provider. When either model returns several detections, the
highest-confidence one is reported.

**Errors** (`400`):

```json
{ "detail": "Unsupported file type. Allowed: ['.jpeg', '.jpg', '.png', '.webp']" }
{ "detail": "Invalid or corrupt image file." }
```

### `GET /health`

```json
{ "status": "ok", "device": "cuda:0", "fallback": "kindwise" }
```

`fallback` is `"kindwise"` when a `KINDWISE_API_KEY` is configured and `"disabled"` when it is not —
a missing key logs a warning at startup but never stops the app booting, since the local model is
the primary provider and works on its own.

## Healthy vs. disease classes

The disease model has 116 classes, mixing actual diseases with healthy-leaf labels (`tomato leaf`, `apple leaf`, `Corn Healthy`, …). `src/services/disease_detection/providers/yolo.py` holds an explicit set of the 33 healthy class IDs.

The set is explicit rather than rule-based on purpose. A rule like `name.endswith(" leaf")` would misclassify `Corn rust leaf` (109) and `Tomato blight leaf` (112) as healthy, and a "contains leaf" rule would break another 25 disease labels such as `corn northern leaf blight` and `tomato leaf mold`. Reporting a blight as healthy is this service's worst failure mode, so the IDs are listed one by one.

Three classes name no pathogen but are not healthy either — `Corn Insects Damages` (99), `Corn Purple Discoloration` (101), `Corn Yellowing` (107). They are deliberately excluded from the healthy set, so they report as the class name with `is_healthy: false`.

## Project structure

```
src/
├── main.py                       # FastAPI app, provider lifecycle at startup, GET /health
├── core/
│   ├── config.py                 # .env configuration
│   └── logging.py                # logger setup
├── models/schema/
│   └── prediction.py             # PredictionResponse
├── services/
│   ├── leaf_detection.py         # load_model(), has_leaf()
│   └── disease_detection/
│       ├── interface.py          # DetectionResult, DiseaseProvider
│       ├── factory.py            # builds the providers, owns the fallback flow
│       └── providers/
│           ├── yolo.py           # YoloProvider, HEALTHY_CLASS_IDS
│           └── kindwise.py       # KindwiseProvider
└── routes/api/v1/
    └── predict.py                # POST /api/v1/predict
```

### Adding or replacing a provider

`interface.py` is the whole contract: a provider is any object with a `name` and an
`async detect(image, raw, content_type)` returning a `DetectionResult` or `None` (`None` meaning
"no usable answer from me"). The arguments are the union of what the providers need — the local
model reads `image`, Kindwise reads `raw` and `content_type` — so each ignores what it does not use.

Escalation policy lives only in `factory.py`; the route holds none of it. Swapping the fallback for a
different vendor means writing one module under `providers/` and changing the import in
`factory.py`. A provider must never raise: `KindwiseProvider.detect` catches its own transport and
parsing failures and returns `None`, which is what keeps a provider outage from breaking the endpoint.

Kindwise is handed the **bytes exactly as uploaded** rather than the decoded `PIL` image, because
re-encoding is known to change the answer (see limitations below).

## Known limitations

- **The leaf gate produces false positives, so `is_plant: true` is weak on its own.** `yolo11x_leaf.pt` detects a "leaf" in a solid blue image at 0.883–0.96 confidence. Raising `LEAF_CONF` does not help at that confidence — it is a limitation of the checkpoint, not the threshold. Kindwise's plant check is the only thing that can overturn it, and it is only consulted when the local disease model is unsure. A confident local detection on a non-plant image will therefore still report `is_plant: true`. In practice the two failures tend to coincide: on that same blue image the disease model returns only a weak `banana leaf` (0.11), which escalates, and Kindwise then correctly returns `is_plant: false`.
- **`is_plant: true` with everything else null is ambiguous.** It covers "the fallback was unreachable or also unsure" and "the local model was unsure and the fallback is disabled". The application log distinguishes them.
- **Re-encoding can change the result.** The same photo saved as `.webp` returned `potato late blight` (0.68) where the JPEG returned `potato early blight` (0.64) — lossy compression is enough to reorder two closely scored classes. Kindwise shows the same sensitivity: the same image sent as webp returned `late blight` (0.525) where the JPEG returned `Alternaria brown spot` (0.605). This is why the fallback receives the original upload bytes untouched.
- The 116-class list contains near-duplicates from two merged training sets (`corn rust` / `Corn rust leaf`, `corn smut` / `Corn Smut`), so the reported class name may vary in capitalization and wording between similar images.
- **Kindwise names do not match the local class names.** The fallback returns its own vocabulary (`Alternaria brown spot`, `late blight`) with different wording and capitalization from the local model's labels (`potato early blight`). Clients that switch on `disease` must handle both vocabularies — `source` tells them which one they got.
- **The fallback is not necessarily more accurate.** On `potato_late.jpeg` the local model reports `potato early blight` (0.64) while Kindwise reports `Alternaria brown spot` (0.605), with `late blight` only second at 0.327. Escalating a weak local detection can replace a correct answer with a confident wrong one; `source` and `confidence` in the response are there to make that measurable on real traffic.
- **The fallback's `is_healthy: true` path is unverified.** Kindwise returns no health flag — a healthy crop appears as an ordinary suggestion — so `HEALTHY_SUGGESTION_NAMES` in `providers/kindwise.py` matches on name. Until it is confirmed against a genuinely healthy leaf, Kindwise can only report a disease or nulls, never a false "healthy".
- **No minimum probability is applied to the fallback's answer.** Its top suggestion is returned whatever its probability, with that probability in `confidence`. Adding a floor is one `if` in `KindwiseProvider._to_result`.
- **Every escalation is a paid API call**, and the service has no upload size limit or rate limit. Junk uploads that get past the leaf gate reach the no-detection path and escalate.
