"""
plate_detector.py
~~~~~~~~~~~~~~~~~
Detects license plate regions in an image frame using a YOLOv8-style
detection model (Hailo8 or ONNX fallback).

Output bounding boxes follow the ``[x1, y1, x2, y2, confidence]`` convention
in *pixel coordinates* of the original (unscaled) input image.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .hailo_inference import create_inference_engine

logger = logging.getLogger(__name__)


@dataclass
class DetectedPlate:
    """A single detected license plate region."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    # Cropped plate image (BGR, set after detection).
    crop: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class PlateDetector:
    """Detect license plate regions in BGR frames.

    Parameters
    ----------
    backend:
        ``"hailo"`` or ``"onnx"``.
    hef_path:
        Path to the YOLOv8 .hef model (Hailo backend).
    onnx_path:
        Path to the YOLOv8 .onnx model (ONNX backend).
    input_width, input_height:
        Model input resolution.
    confidence_threshold:
        Minimum detection confidence to retain.
    nms_threshold:
        IoU threshold for non-maximum suppression.
    min_plate_area:
        Minimum bounding-box area (pixels²) to keep as a valid plate.
    padding:
        Extra pixel border added when cropping the plate from the frame.
    """

    # YOLOv8 output layout: [batch, num_detections, 4 + num_classes]
    # For a single-class (license plate) model the last axis is 5:
    #   [cx, cy, w, h, plate_confidence]
    _CLASS_IDX = 0

    def __init__(
        self,
        backend: str = "onnx",
        hef_path: Optional[str | Path] = None,
        onnx_path: Optional[str | Path] = None,
        input_width: int = 640,
        input_height: int = 640,
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.45,
        min_plate_area: int = 500,
        padding: int = 4,
    ) -> None:
        self._input_w = input_width
        self._input_h = input_height
        self._conf_thresh = confidence_threshold
        self._nms_thresh = nms_threshold
        self._min_area = min_plate_area
        self._padding = padding

        # Only load a model when paths are provided.
        self._engine = None
        if hef_path or onnx_path:
            try:
                self._engine = create_inference_engine(
                    backend=backend,
                    hef_path=hef_path,
                    onnx_path=onnx_path,
                )
            except Exception as exc:
                logger.warning(
                    "Could not load detection model (%s). "
                    "Plate detection will be skipped.",
                    exc,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[DetectedPlate]:
        """Detect license plates in *frame* (BGR).

        Returns a list of :class:`DetectedPlate` sorted by confidence
        (descending).  Each plate's ``crop`` attribute contains the
        cropped BGR region.
        """
        if self._engine is None:
            # No model loaded – treat the entire frame as one plate region.
            logger.debug("No detection model loaded; treating full frame as plate.")
            h, w = frame.shape[:2]
            plate = DetectedPlate(x1=0, y1=0, x2=w, y2=h, confidence=1.0)
            plate.crop = frame.copy()
            return [plate]

        pre, scale_x, scale_y = self._preprocess(frame)
        raw = self._engine.infer(pre)
        plates = self._postprocess(raw, scale_x, scale_y, frame.shape)

        for p in plates:
            p.crop = self._crop_plate(frame, p)

        return sorted(plates, key=lambda p: p.confidence, reverse=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Resize, normalise and transpose the frame for model input.

        Returns
        -------
        (preprocessed_array, scale_x, scale_y)
        """
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self._input_w, self._input_h))
        scale_x = w / self._input_w
        scale_y = h / self._input_h

        # BGR → RGB, HWC → CHW, normalise to [0, 1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        return chw, scale_x, scale_y

    def _postprocess(
        self,
        raw_output: dict,
        scale_x: float,
        scale_y: float,
        original_shape: Tuple[int, ...],
    ) -> List[DetectedPlate]:
        """Parse raw YOLOv8 output into :class:`DetectedPlate` objects."""
        # Grab the first (and typically only) output tensor.
        output = next(iter(raw_output.values()))

        # Shape: (1, num_detections, 5) or (1, 5, num_detections) depending on
        # export settings.  Normalise to (N, 5).
        output = np.squeeze(output)  # remove batch dim
        if output.ndim == 1:
            output = output[np.newaxis, :]
        if output.shape[0] < output.shape[-1]:
            # (5, N) → (N, 5)
            output = output.T

        img_h, img_w = original_shape[:2]
        plates: List[DetectedPlate] = []

        for det in output:
            if det.shape[0] < 5:
                continue
            cx, cy, bw, bh = det[0], det[1], det[2], det[3]
            conf = float(det[4])

            if conf < self._conf_thresh:
                continue

            # Convert normalised cx,cy,w,h → pixel x1,y1,x2,y2.
            x1 = int((cx - bw / 2) * self._input_w * scale_x)
            y1 = int((cy - bh / 2) * self._input_h * scale_y)
            x2 = int((cx + bw / 2) * self._input_w * scale_x)
            y2 = int((cy + bh / 2) * self._input_h * scale_y)

            # Clamp to image bounds.
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            area = (x2 - x1) * (y2 - y1)
            if area < self._min_area:
                continue

            plates.append(DetectedPlate(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf))

        # Apply NMS to suppress overlapping boxes.
        plates = self._apply_nms(plates)
        return plates

    def _apply_nms(self, plates: List[DetectedPlate]) -> List[DetectedPlate]:
        """Non-maximum suppression over detected plates."""
        if not plates:
            return plates

        boxes = np.array([[p.x1, p.y1, p.x2, p.y2] for p in plates], dtype=np.float32)
        scores = np.array([p.confidence for p in plates], dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=self._conf_thresh,
            nms_threshold=self._nms_thresh,
        )

        if len(indices) == 0:
            return []

        # OpenCV returns shape (N, 1) or flat list depending on version.
        flat = (
            indices.flatten().tolist()
            if hasattr(indices, "flatten")
            else list(indices)
        )
        return [plates[i] for i in flat]

    def _crop_plate(self, frame: np.ndarray, plate: DetectedPlate) -> np.ndarray:
        """Return a padded crop of the plate region."""
        h, w = frame.shape[:2]
        x1 = max(0, plate.x1 - self._padding)
        y1 = max(0, plate.y1 - self._padding)
        x2 = min(w, plate.x2 + self._padding)
        y2 = min(h, plate.y2 + self._padding)
        return frame[y1:y2, x1:x2].copy()
