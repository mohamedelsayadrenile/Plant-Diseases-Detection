# Plant Disease Detection API

A small FastAPI service that runs two YOLO models sequentially over an uploaded plant image, escalating to two remote providers when the local model is not confident:

```
Image
  ↓
Leaf YOLO (yolo11x_leaf.pt)  ── no leaf ──→  {"is_plant": false, ...nulls}
  ↓ leaf detected
Disease YOLO (PlantDiseaseDetection.pt)     ── conf ≥ YOLO_DISEASE_CONF ──→  {"source": "yolo", ...}
  ↓ below YOLO_DISEASE_CONF, or nothing detected
Kindwise API                 ── not a plant ──→  {"is_plant": false, "source": "kindwise"}
                             ── prob ≥ KINDWISE_CONF ──→  {"source": "kindwise", ...}
  ↓ below KINDWISE_CONF, failed, or no answer
Gemini (gemini-2.5-flash-lite)              ── answered ──→  {"source": "gemini", ...}
  ↓ unavailable, or also unsure
{"is_plant": true, "disease": null, "is_healthy": null}
```

The leaf model is purely a gate. If it finds no leaf, the disease model never runs. If it passes, the disease model receives the **original uploaded image** — not leaf crops.

The local model is always the primary provider and is always asked first, so a confident local answer never costs an API call. Each subsequent provider is consulted only when the previous one came back below its threshold or with no answer at all. Two rules hold throughout the chain:

- **A rejected answer is never served.** If an escalation ends in silence, the endpoint returns nulls rather than the low-confidence answer that triggered it.
- **A provider failure is never a request failure.** Unreachable, timed out, erroring, or unconfigured providers are logged and skipped; the endpoint still returns 200.

Kindwise's "not a plant" verdict is definitive and ends the chain — it is a real answer, not a weak one, so it does not escalate to the paid model.

Each remote provider is enabled by the presence of its API key and skipped silently (with a startup warning) when it is absent, so the service runs with zero, one, or both of them configured.

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
| `KINDWISE_API_URL` | `https://crop.kindwise.com/api/v1` | Base URL of the second provider |
| `KINDWISE_API_KEY` | *(empty)* | Kindwise API key. Leave empty to disable that provider |
| `KINDWISE_TIMEOUT` | `20.0` | Per-request timeout, in **seconds**, for the Kindwise call |
| `KINDWISE_CONF` | `0.50` | Kindwise answers at or above this probability are trusted; below it Gemini is asked instead |
| `GEMINI_API_KEY` | *(empty)* | Gemini API key. Leave empty to disable that provider |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Model id passed to the Gemini API |
| `GEMINI_TIMEOUT_MS` | `20000` | Per-request timeout, in **milliseconds** — the Gemini SDK takes an int of milliseconds, unlike `KINDWISE_TIMEOUT` above |

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

**Response** — always the same six fields, whichever provider answered:

```json
{
  "is_plant": true,
  "disease": "potato early blight",
  "is_healthy": false,
  "confidence": 0.636,
  "source": "yolo",
  "message": "اللفحة المبكرة في البطاطس. السبب المحتمل: فطر Alternaria solani …\nالعلاج الموصى به: تسميد متوازن …"
}
```

| Case | `is_plant` | `disease` | `is_healthy` | `confidence` | `source` | `message` |
|---|---|---|---|---|---|---|
| Local model confident | `true` | class name | `false` | its score | `"yolo"` | from the sheet |
| Local model confident, healthy class | `true` | `null` | `true` | its score | `"yolo"` | `null` |
| Kindwise answered | `true` | disease name | `false` | its probability | `"kindwise"` | from Gemini |
| Kindwise says not a plant | `false` | `null` | `null` | `null` | `"kindwise"` | `null` |
| Gemini answered | `true` | disease name | `false` | **always `null`** | `"gemini"` | from Gemini |
| Gemini says healthy | `true` | `null` | `true` | `null` | `"gemini"` | `null` |
| Gemini says not a plant | `false` | `null` | `null` | `null` | `"gemini"` | `null` |
| No leaf detected | `false` | `null` | `null` | `null` | `null` | `null` |
| Every provider unavailable or unsure | `true` | `null` | `null` | `null` | `null` | `null` |

`is_plant` is the outermost gate: when it is `false`, every other field is `null`. It is `false` in
exactly two situations — the leaf model found no leaf, or Kindwise's own plant check overruled the
leaf model. Otherwise it is `true`, meaning the leaf gate passed and nothing contradicted it. Note
that a `true` backed only by the leaf gate is weak evidence; see the limitations below.

`source` names the provider that produced the answer and `confidence` is that provider's own score,
so the two are **only comparable within a provider** — YOLO's 0.64 and Kindwise's 0.605 measure
different things. Gemini reports no confidence at all: an LLM's self-assessment is not a calibrated
score, and emitting one would make the weakest answer in the chain look like the other two.
`source: "gemini"` with `confidence: null` is the signal that this answer carries the least
evidence. When a model returns several detections, the highest-confidence one is reported.

`message` is Arabic advice for the grower — the likely cause of the disease, then the recommended
treatment. It is populated **only alongside a named `disease`**: healthy, not-a-plant and
unresolved answers have no advice to give and leave it `null`. See below for where each provider's
text comes from.

**Errors** (`400`):

```json
{ "detail": "Unsupported file type. Allowed: ['.jpeg', '.jpg', '.png', '.webp']" }
{ "detail": "Invalid or corrupt image file." }
```

### `GET /health`

```json
{ "status": "ok", "device": "cuda:0", "providers": ["yolo", "kindwise", "gemini"] }
```

`providers` lists the providers that will actually be tried, in chain order. A remote provider only
appears when its API key is configured; a missing key logs a warning at startup but never stops the
app booting, since the local model is primary and works on its own. So a mis-deployed credential is
visible as a missing entry here.

## Healthy vs. disease classes

The disease model has 116 classes, mixing actual diseases with healthy-leaf labels (`tomato leaf`, `apple leaf`, `Corn Healthy`, …). `src/services/disease_detection/providers/yolo.py` holds an explicit set of the 33 healthy class IDs.

The set is explicit rather than rule-based on purpose. A rule like `name.endswith(" leaf")` would misclassify `Corn rust leaf` (109) and `Tomato blight leaf` (112) as healthy, and a "contains leaf" rule would break another 25 disease labels such as `corn northern leaf blight` and `tomato leaf mold`. Reporting a blight as healthy is this service's worst failure mode, so the IDs are listed one by one.

Three classes name no pathogen but are not healthy either — `Corn Insects Damages` (99), `Corn Purple Discoloration` (101), `Corn Yellowing` (107). They are deliberately excluded from the healthy set, so they report as the class name with `is_healthy: false`.

## The Arabic `message`

The three providers have three label spaces, so the advice is sourced two different ways.

**The local model answers from a closed set**, so its advice is looked up rather than generated:
`src/services/disease_detection/data/disease_messages_ar.json` maps every one of the 116 class
names to its Arabic text. That sheet is the class reference for the local model — every class the
checkpoint can emit has a key, and the 33 healthy ones map to `null`. Editing the wording is a JSON
edit, no code change. `YoloProvider` diffs the sheet against `model.names` at startup and logs a
warning in either direction, so a checkpoint swap that adds a class surfaces as a log line instead
of a silently null message.

**Kindwise and Gemini answer from open vocabularies**, so nothing can be looked up — Gemini writes
their text. For a Gemini answer it comes back in the same vision call, as a field on the response
schema. For a Kindwise answer the factory makes a second, text-only Gemini call
(`GeminiProvider.describe`) on the name Kindwise returned; it trusts that name rather than
re-diagnosing it. Both prompts share one spec (`_ADVICE_SPEC`) that pins the format to the sheet's,
so an answer reads the same to the grower whichever provider produced it.

A missing message is never worth a missing verdict: if Gemini is disabled or the call fails, the
Kindwise answer is still served, with `message: null`.

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
│       ├── factory.py            # builds the providers, owns the chain
│       ├── messages.py           # loads the Arabic advice sheet
│       ├── data/
│       │   └── disease_messages_ar.json   # 116 classes -> Arabic advice
│       └── providers/
│           ├── yolo.py           # YoloProvider, HEALTHY_CLASS_IDS
│           ├── kindwise.py       # KindwiseProvider
│           └── gemini.py         # GeminiProvider
└── routes/api/v1/
    └── predict.py                # POST /api/v1/predict
```

### Adding or replacing a provider

`interface.py` is the whole contract: a provider is any object with a `name` and an
`async detect(image, raw, content_type)` returning a `DetectionResult` or `None` (`None` meaning
"no usable answer from me"). The arguments are the union of what the providers need — the local
model reads `image`, Kindwise reads `raw` and `content_type` — so each ignores what it does not use.

Escalation policy lives only in `factory.py`; the route holds none of it, and adding the third
provider required no change to `interface.py`, `predict.py`, or the response schema. Adding or
swapping a vendor means writing one module under `providers/` and wiring it into `factory.startup()`
and the chain.

A provider must never raise. `KindwiseProvider.detect` and `GeminiProvider.detect` each catch their
own transport and parsing failures and return `None`, which is what keeps a provider outage from
breaking the endpoint.

Both remote providers are handed the **bytes exactly as uploaded** rather than the decoded `PIL`
image, because re-encoding is known to change the answer (see limitations below). Every allowed
extension — jpeg, png, webp — is natively accepted by both APIs, so nothing is ever re-encoded.

Two details of the Gemini SDK are easy to get wrong and are commented at their use sites: its
timeout is an **int of milliseconds** (not seconds like Kindwise's), and `response.parsed` is
silently `None` — no exception, no log — when a reply is blocked, empty, or fails schema validation,
so it is checked explicitly.

## Known limitations

- **The leaf gate produces false positives, so `is_plant: true` is weak on its own.** `yolo11x_leaf.pt` detects a "leaf" in a solid blue image at 0.883–0.96 confidence. Raising `LEAF_CONF` does not help at that confidence — it is a limitation of the checkpoint, not the threshold. Kindwise's plant check is the only thing that can overturn it, and it is only consulted when the local disease model is unsure. A confident local detection on a non-plant image will therefore still report `is_plant: true`. In practice the two failures tend to coincide: on that same blue image the disease model returns only a weak `banana leaf` (0.11), which escalates, and Kindwise then correctly returns `is_plant: false`.
- **`is_plant: true` with everything else null is ambiguous.** It covers "the fallback was unreachable or also unsure" and "the local model was unsure and the fallback is disabled". The application log distinguishes them.
- **Re-encoding can change the result.** The same photo saved as `.webp` returned `potato late blight` (0.68) where the JPEG returned `potato early blight` (0.64) — lossy compression is enough to reorder two closely scored classes. Kindwise shows the same sensitivity: the same image sent as webp returned `late blight` (0.525) where the JPEG returned `Alternaria brown spot` (0.605). This is why the fallback receives the original upload bytes untouched.
- The 116-class list contains near-duplicates from two merged training sets (`corn rust` / `Corn rust leaf`, `corn smut` / `Corn Smut`), so the reported class name may vary in capitalization and wording between similar images.
- **Three providers means three vocabularies.** YOLO returns its 116 class labels (`potato early blight`), Kindwise its own (`Alternaria brown spot`, `late blight`), and Gemini is unconstrained free text, so the same disease can come back worded three ways. Clients that switch on `disease` must key off `source`. Constraining Gemini to the local label space is possible — the 116 class names are recoverable from the checkpoint — if that ever becomes preferable to free-form naming.
- **A later provider is not necessarily more accurate than an earlier one.** On `potato_late.jpeg` the local model reports `potato early blight` (0.64) while Kindwise reports `Alternaria brown spot` (0.605), with `late blight` only second at 0.327. Escalating a weak detection can replace a correct answer with a confident wrong one; `source` and `confidence` are there to make that measurable on real traffic.
- **Gemini answers where the specialised models declined to.** It is reached exactly when the evidence is weakest, and it returns no calibrated score, so a confident-sounding disease name arrives with nothing to weigh it against. Treat `source: "gemini"` as the lowest-evidence tier of answer.
- **Kindwise's `is_healthy: true` path is unverified.** Kindwise returns no health flag — a healthy crop appears as an ordinary suggestion — so `HEALTHY_SUGGESTION_NAMES` in `providers/kindwise.py` matches on name. Until it is confirmed against a genuinely healthy leaf, Kindwise can only report a disease or nulls, never a false "healthy".
- **`KINDWISE_CONF = 0.50` is calibrated from two observed samples** (0.605 for the jpeg, 0.525 for the same image as webp), both of which only just clear it. A small dip in image quality can flip an answer from "accepted" to "escalated to the paid model", so watch the escalation rate in the logs.
- **Cost compounds, and nothing bounds it.** A single upload can now bill two paid APIs, and a Kindwise-served answer bills Gemini too for its `message`. The service still has no upload size limit and no rate limit, and Gemini additionally hard-fails above a 20 MB total request — a size an unbounded upload can reach. Junk uploads that get past the leaf gate reach the no-detection path and escalate through the whole chain.
- **A Kindwise answer costs an extra Gemini round trip.** `describe` is a second serial call after Kindwise returns, so that branch's latency is Kindwise plus Gemini rather than Kindwise alone. It is only paid on the Kindwise-served path — the local model looks its advice up for free, and Gemini writes its own in the call it was already making.
- **Only the local model's advice is reviewed text.** The sheet is fixed and auditable; the Kindwise and Gemini messages are generated per request, so their wording — and the active ingredients they name — varies between calls and is not reviewed before it reaches the grower.
- **`generate_content` is Google's legacy surface.** The SDK still fully supports it and `gemini-2.5-flash-lite` is GA with no announced shutdown date, but Google is steering new work toward `client.interactions.create` and the Gemini 3.x models, so `providers/gemini.py` will need revisiting.
