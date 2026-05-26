# Models for ANP-Recognition

This directory is where you place the compiled **Hailo Executable Format (.hef)**
and/or **ONNX** model files required by the pipeline.

---

## Required models

| Role | Hailo HEF | ONNX fallback |
|------|-----------|---------------|
| License-plate detector | `yolov8n_license_plate.hef` | `yolov8n_license_plate.onnx` |
| License-plate OCR (text reader) | `lprnet.hef` | `lprnet.onnx` |

---

## Obtaining Hailo HEF models

1. Register for a free account at the [Hailo Developer Zone](https://hailo.ai/developer-zone/).
2. Navigate to **AI Zoo → Model Zoo** and search for:
   - **YOLOv8n** (vehicle / license plate detection)
   - **LPRNet** (license plate recognition / OCR)
3. Download the pre-compiled `.hef` files and copy them to this directory.

Alternatively you can compile your own models using the
[Hailo Dataflow Compiler](https://hailo.ai/developer-zone/documentation/) with the
`hailo compile` CLI.

---

## Obtaining ONNX fallback models

ONNX models are used when Hailo hardware is not available (`backend: onnx` in
`config/config.yaml`).

* **YOLOv8n for license plates** – export from
  [Ultralytics YOLOv8](https://docs.ultralytics.com/modes/export/):
  ```bash
  yolo export model=yolov8n.pt format=onnx imgsz=640
  ```
  Then fine-tune on a license plate dataset such as
  [Open Images V7 – Vehicle Registration Plates](https://storage.googleapis.com/openimages/web/index.html).

* **LPRNet** – the reference implementation is available at
  [sirius-ai/LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch).
  Export the trained weights with `torch.onnx.export`.

---

## Directory layout after setup

```
models/
├── README.md
├── yolov8n_license_plate.hef   # Hailo8 detector model
├── yolov8n_license_plate.onnx  # ONNX detector model (CPU fallback)
├── lprnet.hef                  # Hailo8 OCR model
└── lprnet.onnx                 # ONNX OCR model (CPU fallback)
```
