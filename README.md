# Gesture Engine

A real-time hand gesture recognition system built with Python, MediaPipe, OpenCV, scikit-learn, Streamlit, WebRTC, and Docker.

The project focuses on low-latency gesture recognition, stable predictions, a polished browser UI, and reliable webhook-based event integration.

## 1. Overview

Gesture Engine detects predefined hand gestures from a laptop webcam, displays the live result in a browser-based Streamlit interface, and can forward stable gesture-state events to an external system through a configurable HTTP webhook.

The production UI uses browser WebRTC rather than a blocking OpenCV camera loop. The video processor runs the existing detection, feature extraction, prediction, rendering, and webhook components on the WebRTC worker.

The project was designed to satisfy the project requirements, including webcam access, browser UI, real-time detection, webhook integration, repeated-trigger prevention, Git version control, Docker containerization, and documentation. fileciteturn3file0

---

## 2. Features

- Real-time browser webcam processing
- WebRTC video transport
- MediaPipe Hand Landmarker
- 21 hand landmarks
- 63-dimensional landmark representation
- SVM production classifier
- Decision Tree evaluated during model development
- Probability-based confidence
- 70% confidence threshold
- Seven-sample temporal smoothing
- `UNKNOWN` handling for low-confidence predictions
- OpenCV landmark/status rendering
- 10 supported gestures
- Configurable HTTP/HTTPS webhook
- JSON gesture events
- JSON no-hand events
- Duplicate event suppression
- Asynchronous webhook delivery
- Webhook/network failures isolated from CV processing
- 640×360 processing limit
- Approximately 24 FPS browser input target
- SVM inference every third frame
- Dockerized deployment

---

## 3. Requirements

| Assignment requirement | Gesture Engine implementation |
|---|---|
| Laptop webcam | Browser camera through WebRTC |
| Browser-based interface | Streamlit |
| Real-time detection | MediaPipe + feature extractor + SVM |
| At least five gestures | 10 gestures |
| Visual detection output | OpenCV overlay |
| Configurable webhook URL | Streamlit input |
| Gesture webhook | HTTP POST JSON |
| No-hand handling | `hand_not_detected` event |
| Invalid webhook handling | HTTP/HTTPS validation |
| Webhook failure handling | Exceptions contained by webhook client |
| Repeated-trigger prevention | State/event suppression |
| Python | Python 3.14.x |
| Git | Git repository |
| Docker | Dockerfile + container |
| Documentation | This README |

---

# 4. Architecture

```text
Browser Webcam
      ↓
    WebRTC
      ↓
 Hand Detection
  (MediaPipe)
      ↓
 21 Landmarks
      ↓
63 Features
      ↓
 SVM Predictor
      ↓
Confidence Threshold
      ↓
Temporal Smoothing
      ↓
 Stable Gesture
    ↙       ↘
Render      Webhook
(OpenCV)   (JSON POST)
```

The processing path is intentionally linear: the browser supplies the live video through WebRTC, MediaPipe extracts hand landmarks, the feature extractor produces the classifier input, and the predictor produces a confidence-aware temporally smoothed result. The resulting stable state is then used independently for visual rendering and external webhook delivery.

# 5. Repository Structure

```text
Gesture Engine/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── webhooks.py
├── .gitignore
│
├── camera/
│   └── ...
│
├── detection/
│   └── ...
│
├── features/
│   └── ...
│
├── models/
│   ├── predictor.py
│   └── trained/
│       └── svm_model.joblib
│
├── scripts/
│   └── ...
│
├── visualization/
│   └── ...
│
├── ui/
│   └── streamlit_app.py
│
└── data/
    └── ...
```

## Module responsibilities

### `app.py`

Application-side/native entry point for the core project pipeline.

### `camera/`

Camera functionality used by the native/OpenCV application path.

The production browser UI uses browser WebRTC instead of directly opening the webcam from Python.

### `detection/`

Contains `HandDetector`, responsible for MediaPipe hand detection and landmark extraction.

### `features/`

Contains `LandMarkFeatureExtractor`, responsible for converting the detected hand landmarks into the classifier feature representation.

### `models/`

Contains the prediction layer.

`models/predictor.py` defines `GesturePredictor`.

The predictor loads:

```text
models/trained/svm_model.joblib
```

and performs probability-based classification and temporal smoothing.

### `models/trained/`

Contains trained inference artifacts.

### `scripts/`

Contains project development utilities for data/model workflows.

### `visualization/`

Contains `HandRenderer`, responsible for drawing hand landmarks/skeleton information on frames.

### `ui/`

Contains the production browser UI:

```text
ui/streamlit_app.py
```

It combines Streamlit, WebRTC, OpenCV, the existing CV/ML components, and webhook integration.

### `webhooks.py`

Independent external-system integration layer.

It handles:

- endpoint configuration
- JSON serialization
- asynchronous HTTP POST
- gesture events
- no-hand events
- duplicate-event suppression
- network/HTTP failure isolation

---

# 6. Computer Vision Pipeline

## 6.1 WebRTC camera capture

The browser owns camera capture.

```text
Laptop Camera
     ↓
Browser
     ↓
WebRTC
     ↓
Python WebRTC processor
```

This avoids putting a blocking `while True` OpenCV capture loop inside Streamlit.

## 6.3 Hand detection

`HandDetector` uses MediaPipe Hand Landmarker.

The detected hand contains 21 landmarks.

## 6.4 Feature extraction

Each landmark contributes:

```text
x + y + z
```

Therefore:

```text
21 landmarks × 3 coordinates = 63 features
```

The feature extractor converts the landmark representation into the vector consumed by the trained classifier.

---

# 7. Machine Learning

## SVM

The production model is an SVM stored at:

```text
models/trained/svm_model.joblib
```

`GesturePredictor` loads this artifact with `joblib`.

Prediction uses:

```text
predict_proba()
```

The class with the highest probability becomes the prediction candidate.

## Decision Tree

A Decision Tree was evaluated during model development as an alternative/baseline classifier.

The final production predictor uses the SVM because the project selected it as the production classifier.

---

# 8. Confidence and Temporal Smoothing

The current predictor configuration is:

```python
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 7
```

The prediction process is:

```text
Feature vector
     ↓
SVM probabilities
     ↓
Highest probability class
     ↓
Confidence check
     │
     ├── confidence < 0.70 → UNKNOWN
     │                         + clear history
     │
     └── confidence >= 0.70
                    ↓
             prediction history
                    ↓
              stable gesture
```

The seven-sample history reduces frame-to-frame classification instability.

The predictor returns:

```text
gesture, confidence
```

where the gesture can be a supported class or `UNKNOWN`.

---

# 9. Supported Gestures

The current application supports:

| Gesture | Icon |
|---|---|
| `PINCH` | 🤏 |
| `ITALIAN` | 🤌 |
| `ILY` | 🤟 |
| `ROCK` | 🤘 |
| `CALL_ME` | 🤙 |
| `FINGER_HEART` | 🫰 |
| `OPEN_PALM` | 🖐️ |
| `FIST` | ✊ |
| `ONE` | ☝️ |
| `TWO` | ✌️ |

---

# 10. Real-Time Performance Design

The production WebRTC processor uses several deliberate optimizations.

### Components are initialized once per stream

The processor creates:

```text
HandDetector
LandMarkFeatureExtractor
GesturePredictor
HandRenderer
WebhookClient
```

once when the stream starts.

They are not constructed for every frame. fileciteturn4file0

### Inference throttling

```text
INFER_EVERY_N_FRAMES = 3
```

The feature extraction + prediction stage runs every third frame.

### Skeleton rendering

The hand landmarks can still be rendered on every received frame, while the latest prediction is reused between inference frames.

### No per-frame Streamlit rerun

The production UI uses WebRTC's video processor instead of a Streamlit polling/capture loop.

This keeps the real-time processing path separate from ordinary Streamlit UI execution.

---

# 11. Streamlit UI

The production interface is:

```text
ui/streamlit_app.py
```

It provides:

- live recognition
- camera status
- model information
- supported gestures
- webhook configuration
- webhook endpoint status
- processing pipeline information
- architecture information
- performance information

The UI uses a dark dashboard-style design and exposes the webhook endpoint without hardcoding it.

---

# 12. Webhook Integration

## Configuration

Enter an HTTP or HTTPS endpoint into the Webhook section.

Example:

```text
https://example.com/webhook
```

For testing, a temporary endpoint such as Webhooks.site can be used.

## Gesture events

A stable gesture is sent as:

```json
{
  "event": "gesture_detected",
  "gesture": "ILY",
  "confidence": 0.94,
  "timestamp": "2026-08-30T00:00:00+00:00"
}
```

## No-hand events

When the hand disappears:

```json
{
  "event": "hand_not_detected",
  "gesture": null,
  "confidence": 0.0,
  "timestamp": "2026-08-30T00:00:00+00:00"
}
```

## Event suppression

Webhook delivery is state based rather than frame based.

For example:

```text
ILY
ILY
ILY
ILY
```

produces one gesture event.

Then:

```text
ILY → ROCK
```

produces one new gesture event.

And:

```text
ROCK → NO HAND
```

produces one `hand_not_detected` event.

Continuous no-hand frames do not generate continuous requests.

## Asynchronous delivery

The webhook request is dispatched from a daemon background thread.

This prevents a slow external endpoint from blocking the WebRTC video-processing path.

HTTP and network exceptions are caught inside the webhook module so webhook failures do not terminate the recognition stream.

---

# 13. Local Installation

## Requirements

- Python 3.14.x
- Laptop webcam
- Modern browser
- Git
- Internet access for package installation

The development environment uses Python 3.14.6.

## Clone

```bash
git clone https://github.com/Vijay-Nagadamudi/Gesture-Engine.git
cd "Gesture Engine"
```

## Create virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

## Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 14. Run Locally

Start the production UI:

```powershell
python -m streamlit run ui/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

Allow browser camera access.

Start the WebRTC stream and perform one of the supported gestures.

---

# 15. Webhook Testing

1. Open a temporary webhook receiver.
2. Copy its generated URL.
3. Start Gesture Engine.
4. Paste the URL into the Webhook field.
5. Start the camera.
6. Perform a gesture.
7. Inspect the received JSON.
8. Hold the same gesture and confirm requests do not continuously repeat.
9. Remove the hand.
10. Confirm one `hand_not_detected` event arrives.
11. Perform another gesture and confirm a new event arrives.

---

# 16. Docker

The project includes:

```text
Dockerfile
```

The image uses:

```text
python:3.14-slim
```

The Dockerfile also installs native Linux runtime libraries required by the OpenCV/MediaPipe/WebRTC stack before installing Python dependencies.

## Build

From the project root:

```powershell
docker build -t gesture-engine .
```

## Run

```powershell
docker run --rm -p 8501:8501 gesture-engine
```

Open:

```text
http://localhost:8501
```

Allow browser camera access.

The containerized application provides the same core browser-based recognition and webhook functionality as the local implementation.

---

# 17. Docker Architecture

```text
Host Machine
     │
     ├── Browser
     │     └── Webcam
     │
     ▼
localhost:8501
     │
     ▼
┌───────────────────────────────┐
│        Docker Container       │
│                               │
│ Streamlit                     │
│      │                        │
│ WebRTC Video Processor        │
│      │                        │
│ MediaPipe                     │
│      │                        │
│ Feature Extractor             │
│      │                        │
│ SVM Predictor                 │
│      │                        │
│ OpenCV Renderer               │
│      │                        │
│ Webhook Client                │
└───────────────────────────────┘
```

The webcam is captured by the browser rather than requiring direct host-camera device mapping into the Linux container.

---

# 18. Python Dependency Stack

The repository pins its complete Python dependency environment in `requirements.txt`.

Major packages include:

### Computer vision

```text
opencv-python
opencv-contrib-python
mediapipe
```

### Machine learning

```text
scikit-learn
scipy
numpy
joblib
```

### UI

```text
streamlit
streamlit-webrtc
```

### Media/WebRTC

```text
av
aiortc
aioice
pylibsrtp
websockets
```

### Data and supporting libraries

```text
pandas
matplotlib
pillow
protobuf
requests
python-dateutil
packaging
```

Additional pinned transitive dependencies are maintained in `requirements.txt`.

---

# 18. Error Handling

### Invalid webhook URL

The UI accepts configured HTTP/HTTPS endpoints and warns when the entered value is not a valid HTTP/HTTPS URL.

### Webhook failure

The webhook module catches HTTP/network failures.

The recognition pipeline continues running.

### No hand

The UI displays:

```text
NO HAND
```

and the integration can send:

```text
hand_not_detected
```

### Low confidence

The predictor returns:

```text
UNKNOWN
```

when the highest class probability is below `0.70`.

### Camera access

Camera capture is browser controlled. If the camera cannot start, browser/device permissions and competing camera applications must be checked.

---

# 19. Configuration Values

## Predictor

```text
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 7
```

## WebRTC

```text
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
VIDEO_FPS = 24
INFER_EVERY_N_FRAMES = 3
```

## Webhook

The webhook URL is supplied through the UI rather than embedded as a fixed endpoint.

---

# 20. Security and Git Hygiene

Do not commit:

- `.venv/`
- private webhook URLs
- API keys
- passwords
- authentication tokens
- generated cache files
- unnecessary local artifacts

The repository contains `.gitignore` for generated and environment-specific files.

Webhook endpoints should be treated as configuration and should not be hardcoded with secrets.

---

# 21. Assumptions and Limitations

### Single primary hand

The current production processor extracts the first detected hand for classification.

### Model scope

The classifier recognizes only the gesture classes represented by its training data. It is not a general-purpose sign-language recognition system.

### Environment

Recognition quality can vary with:

- lighting
- camera quality
- hand visibility
- camera distance
- hand orientation
- gesture execution
- similarity to training samples

### CPU-oriented design

The architecture is intended to operate on a standard laptop and does not require a dedicated GPU for the intended workload.

### Network requirement

Webhook delivery requires a reachable HTTP/HTTPS endpoint.

### Browser permissions

The production UI requires browser camera permission.

---

# 22. Verification Checklist

### Application
- [ ] Streamlit UI loads
- [ ] WebRTC camera starts
- [ ] Hand detection and gesture recognition work
- [ ] Confidence and `UNKNOWN` handling work

### Webhook
- [ ] Gesture JSON is received
- [ ] `hand_not_detected` JSON is received
- [ ] Repeated states do not spam requests
- [ ] Gesture changes generate new events
- [ ] Webhook failures do not stop recognition

### Docker
- [ ] Image builds successfully
- [ ] Container starts on port 8501
- [ ] MediaPipe initializes inside the container
- [ ] Browser camera works
- [ ] Gesture recognition and webhook integration work

### Repository
- [ ] `Dockerfile`, `requirements.txt`, `webhooks.py`, and README are committed
- [ ] `.venv` and secrets are excluded

# 23. GitHub

Repository:

https://github.com/Vijay-Nagadamudi/Gesture-Engine

---

# 26. Final Architecture Summary

```text
                 GESTURE ENGINE
                       │
                       ▼
                Browser Webcam
                       │
                       ▼
                    WebRTC
                       │
                       ▼
                Hand Detection
                  MediaPipe
                       │
                       ▼
                 21 Landmarks
                       │
                       ▼
                  63 Features
                       │
                       ▼
                     SVM
                       │
                       ▼
             Confidence Threshold
                       │
                       ▼
             Temporal Smoothing
                Window = 7
                       │
                       ▼
               Stable Prediction
                  │          │
                  │          └────────── No Hand
                  │                       │
                  ▼                       ▼
             Gesture Event          No-Hand Event
                  │                       │
                  └──────────┬────────────┘
                             ▼
                       Webhook Client
                             │
                       Async HTTP POST
                             │
                             ▼
                      External System
```

Gesture Engine separates real-time computer vision, machine-learning inference, rendering, UI transport, and external event delivery into distinct responsibilities.

The result is a browser-based, low-latency gesture recognition application with reproducible Docker deployment and configurable external-system integration.
