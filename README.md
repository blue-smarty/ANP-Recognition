# ANP-Recognition

Australian Number Plate Recognition (ANPR) system that uses either still
images or live camera streams to detect and read licence plates from all
Australian states and territories.  Inference is accelerated by the
[Hailo8](https://hailo.ai/products/hailo-8/) edge-AI chip, with automatic
CPU fallback via ONNX-Runtime and Tesseract OCR for development and testing.

---

## Features

| Capability | Details |
|---|---|
| **Input sources** | Still images (JPG, PNG, BMP, …), video files, webcam, RTSP / HTTP streams |
| **Plate detection** | YOLOv8n compiled to HEF for Hailo8 (ONNX fallback) |
| **Text recognition** | LPRNet compiled to HEF for Hailo8 (ONNX / Tesseract fallback) |
| **State classification** | All 8 jurisdictions: NSW, VIC, QLD, SA, WA, TAS, ACT, NT |
| **Annotation** | Bounding box, plate text, state label and confidence overlay |
| **Output** | Live window and/or saved annotated images |

---

## Supported Australian States and Territories

| Code | Jurisdiction | Example plate |
|------|---|---|
| NSW | New South Wales | `ABC-12D` |
| VIC | Victoria | `1AB-2CD` |
| QLD | Queensland | `123-ABC` / `ABC-12A` |
| SA | South Australia | `ABC-123` |
| WA | Western Australia | `1ABC-234` |
| TAS | Tasmania | `AB-12-CD` |
| ACT | Australian Capital Territory | `YAB-00A` |
| NT | Northern Territory | `CA-12-3B` |

---

## Requirements

* Python 3.9 or later
* OpenCV 4.8+
* A Hailo8 device + HailoRT (optional – ONNX/Tesseract used as fallback)
* Tesseract OCR binary (optional – used when no neural OCR model is loaded)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For OpenCV GUI windows, recent Qt builds expect fonts to come from the host
system or your app bundle. This repository ships a minimal DejaVu Sans set in
`src/anpr/fonts/` and configures `QT_QPA_FONTDIR` automatically before OpenCV
loads. If you repackage the application, keep that directory with the app or
install `fontconfig` plus a DejaVu font package such as `fonts-dejavu-core`.

### Tesseract (optional fallback OCR)

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows – download the installer from
# https://github.com/UB-Mannheim/tesseract/wiki
```

### Hailo Runtime (optional – for Hailo8 hardware)

Download and install the HailoRT wheel from the
[Hailo Developer Zone](https://hailo.ai/developer-zone/).

```bash
pip install hailort-*.whl
```

---

## Getting the Models

See [models/README.md](models/README.md) for instructions on downloading the
pre-compiled Hailo HEF and ONNX model files.

---

## Usage

### Process a still image

```bash
python main.py image path/to/plate.jpg
```

### Process a directory of images

```bash
python main.py image path/to/images/
```

### Live webcam (default device 0)

```bash
python main.py camera
```

### Specific camera index or video file

```bash
python main.py camera --source 1
python main.py camera --source /path/to/video.mp4
```

### RTSP / HTTP stream

```bash
python main.py camera --source "rtsp://user:pass@192.168.1.100/stream"
```

### Save annotated output

```bash
python main.py image plate.jpg --save-dir results/
python main.py camera --save-dir results/
```

### Headless (no display window)

```bash
python main.py image plate.jpg --no-window
python main.py camera --source 0 --no-window
```

### Custom configuration

```bash
python main.py --config path/to/custom_config.yaml image plate.jpg
```

Press **q** or **Esc** to quit the live camera window.

---

## Configuration

Edit `config/config.yaml` to change backend, model paths, thresholds and
display settings.  Key options:

```yaml
inference:
  backend: hailo        # "hailo" or "onnx"
  hailo:
    detector_hef: models/yolov8n_license_plate.hef
    recognizer_hef: models/lprnet.hef
  onnx:
    detector_onnx: models/yolov8n_license_plate.onnx
    recognizer_onnx: models/lprnet.onnx

detector:
  confidence_threshold: 0.45
  nms_threshold: 0.45
```

---

## Project Structure

```
ANP-Recognition/
├── main.py                    # CLI entry point
├── requirements.txt
├── config/
│   └── config.yaml            # Runtime configuration
├── models/
│   └── README.md              # Model download instructions
├── src/
│   └── anpr/
│       ├── __init__.py
│       ├── hailo_inference.py  # Hailo8 / ONNX inference engine
│       ├── plate_detector.py   # YOLOv8 plate detection
│       ├── plate_recognizer.py # LPRNet / Tesseract OCR
│       ├── state_classifier.py # Australian state identification
│       ├── pipeline.py         # Orchestration pipeline
│       └── utils.py            # Drawing, I/O helpers
└── tests/
    └── test_pipeline.py        # Unit tests
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Architecture

```
Input (image / camera frame)
        │
        ▼
┌───────────────────┐
│   PlateDetector   │  YOLOv8 → bounding boxes
│  (Hailo8 / ONNX)  │
└────────┬──────────┘
         │  cropped plate regions
         ▼
┌───────────────────┐
│  PlateRecognizer  │  LPRNet → text string
│ (Hailo8/ONNX/Tess)│
└────────┬──────────┘
         │  "ABC12D"
         ▼
┌───────────────────────────┐
│  AustralianStateClassifier│  regex + colour → state
└────────┬──────────────────┘
         │  NSW / VIC / QLD / …
         ▼
  Annotated output frame
```

---

## License

MIT
