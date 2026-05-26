"""
pipeline.py
~~~~~~~~~~~
High-level ANPR orchestration pipeline.

Usage example::

    from src.anpr import ANPRPipeline

    pipeline = ANPRPipeline.from_config("config/config.yaml")

    # Single image
    results = pipeline.process_image("photo.jpg")
    for r in results:
        print(r.plate_text, r.state.value, r.confidence)

    # Camera / video stream
    pipeline.run_stream(source=0)  # 0 = default webcam
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import yaml

from .plate_detector import PlateDetector, DetectedPlate
from .plate_recognizer import PlateRecognizer
from .state_classifier import AustralianState, AustralianStateClassifier, StateClassification
from .utils import draw_plate_annotation, load_image, open_camera, save_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlateResult:
    """Recognition result for a single detected plate."""

    plate_text: str
    state: AustralianState
    state_name: str
    detection_confidence: float
    state_confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    crop: Optional[np.ndarray] = field(default=None, repr=False)

    def __str__(self) -> str:
        return (
            f"Plate: {self.plate_text or '(unread)'}  "
            f"State: {self.state_name}  "
            f"DetConf: {self.detection_confidence:.0%}  "
            f"StateConf: {self.state_confidence:.0%}"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ANPRPipeline:
    """End-to-end Australian Number Plate Recognition pipeline.

    Parameters
    ----------
    detector:
        Configured :class:`PlateDetector` instance.
    recognizer:
        Configured :class:`PlateRecognizer` instance.
    classifier:
        Configured :class:`AustralianStateClassifier` instance.
    show_window:
        Display annotated frames in an OpenCV window.
    window_name:
        OpenCV window title.
    frame_delay_ms:
        ``cv2.waitKey`` delay between video frames.
    save_dir:
        Directory for saving annotated output images.  ``None`` disables saving.
    """

    def __init__(
        self,
        detector: PlateDetector,
        recognizer: PlateRecognizer,
        classifier: AustralianStateClassifier,
        show_window: bool = True,
        window_name: str = "ANP-Recognition",
        frame_delay_ms: int = 1,
        save_dir: Optional[str | Path] = None,
    ) -> None:
        self._detector = detector
        self._recognizer = recognizer
        self._classifier = classifier
        self._show_window = show_window
        self._window_name = window_name
        self._frame_delay = frame_delay_ms
        self._save_dir = Path(save_dir) if save_dir else None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path = "config/config.yaml") -> "ANPRPipeline":
        """Construct a pipeline from a YAML configuration file."""
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        with cfg_path.open() as fh:
            cfg = yaml.safe_load(fh)

        backend = cfg["inference"]["backend"]
        inf_cfg = cfg["inference"]
        det_cfg = cfg["detector"]
        rec_cfg = cfg["recognizer"]
        disp_cfg = cfg["display"]
        out_cfg = cfg["output"]

        detector = PlateDetector(
            backend=backend,
            hef_path=inf_cfg["hailo"]["detector_hef"] if backend == "hailo" else None,
            onnx_path=inf_cfg["onnx"]["detector_onnx"] if backend == "onnx" else None,
            input_width=det_cfg["input_width"],
            input_height=det_cfg["input_height"],
            confidence_threshold=det_cfg["confidence_threshold"],
            nms_threshold=det_cfg["nms_threshold"],
            min_plate_area=det_cfg["min_plate_area"],
            padding=rec_cfg["padding"],
        )

        recognizer = PlateRecognizer(
            backend=backend,
            hef_path=inf_cfg["hailo"]["recognizer_hef"] if backend == "hailo" else None,
            onnx_path=inf_cfg["onnx"]["recognizer_onnx"] if backend == "onnx" else None,
            input_width=rec_cfg["input_width"],
            input_height=rec_cfg["input_height"],
            tesseract_config=rec_cfg["tesseract_config"],
        )

        classifier = AustralianStateClassifier()

        return cls(
            detector=detector,
            recognizer=recognizer,
            classifier=classifier,
            show_window=bool(disp_cfg.get("window_name", "")),
            window_name=disp_cfg.get("window_name", "ANP-Recognition"),
            frame_delay_ms=disp_cfg["frame_delay_ms"],
            save_dir=out_cfg.get("save_dir") or None,
        )

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def process_image(
        self,
        image: Union[str, Path, np.ndarray],
        annotate: bool = True,
    ) -> List[PlateResult]:
        """Run the full pipeline on a single image.

        Parameters
        ----------
        image:
            File path or a BGR NumPy array.
        annotate:
            Draw detection results on the image and optionally display / save it.

        Returns
        -------
        List of :class:`PlateResult`, one per detected plate.
        """
        if isinstance(image, (str, Path)):
            frame = load_image(image)
            if frame is None:
                return []
            source_path = Path(image)
        else:
            frame = image
            source_path = None

        results = self._process_frame(frame)

        if annotate:
            annotated = self._annotate_frame(frame.copy(), results)
            if self._show_window:
                cv2.imshow(self._window_name, annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if self._save_dir and source_path:
                out_path = self._save_dir / source_path.name
                save_image(annotated, out_path)

        for r in results:
            logger.info(str(r))

        return results

    # ------------------------------------------------------------------
    # Video / camera stream
    # ------------------------------------------------------------------

    def run_stream(
        self,
        source: Union[int, str] = 0,
        max_frames: Optional[int] = None,
        annotate: bool = True,
    ) -> None:
        """Process a live camera feed or video file.

        Parameters
        ----------
        source:
            Camera index (``int``) or file/stream path (``str``).
        max_frames:
            Stop after this many frames.  *None* runs indefinitely.
        annotate:
            Overlay detection results on displayed frames.
        """
        cap = open_camera(source)
        frame_idx = 0
        fps_start = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.info("End of video stream.")
                    break

                results = self._process_frame(frame)

                if annotate:
                    display = self._annotate_frame(frame, results)
                    # FPS overlay.
                    elapsed = time.time() - fps_start
                    fps = (frame_idx + 1) / max(elapsed, 1e-6)
                    cv2.putText(
                        display, f"FPS: {fps:.1f}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 200, 255), 2, cv2.LINE_AA,
                    )
                    if self._show_window:
                        cv2.imshow(self._window_name, display)

                    if self._save_dir:
                        save_image(display, self._save_dir / f"frame_{frame_idx:06d}.jpg")

                # Log to console.
                for r in results:
                    if r.plate_text:
                        logger.info("Frame %d  %s", frame_idx, r)

                if self._show_window:
                    key = cv2.waitKey(self._frame_delay) & 0xFF
                    if key == ord("q") or key == 27:  # q or ESC to quit
                        logger.info("User requested quit.")
                        break
                elif self._frame_delay > 0:
                    # Keep a similar pacing in headless mode without invoking GUI APIs.
                    time.sleep(self._frame_delay / 1000.0)

                frame_idx += 1
                if max_frames is not None and frame_idx >= max_frames:
                    logger.info("Reached max_frames=%d", max_frames)
                    break

        finally:
            cap.release()
            if self._show_window:
                cv2.destroyAllWindows()
            logger.info("Stream closed after %d frames.", frame_idx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> List[PlateResult]:
        """Run detection → OCR → state classification on one frame."""
        detected_plates: List[DetectedPlate] = self._detector.detect(frame)
        results: List[PlateResult] = []

        for plate in detected_plates:
            text = self._recognizer.recognize(plate.crop)
            cls_result: StateClassification = self._classifier.classify(
                text, plate_crop=plate.crop
            )
            results.append(
                PlateResult(
                    plate_text=text,
                    state=cls_result.state,
                    state_name=self._classifier.state_display_name(cls_result.state),
                    detection_confidence=plate.confidence,
                    state_confidence=cls_result.confidence,
                    bbox=plate.bbox,
                    crop=plate.crop,
                )
            )

        return results

    def _annotate_frame(
        self, frame: np.ndarray, results: List[PlateResult]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on *frame* (in-place)."""
        for r in results:
            x1, y1, x2, y2 = r.bbox
            draw_plate_annotation(
                frame, x1, y1, x2, y2,
                text=r.plate_text,
                state=r.state.value,
                confidence=r.detection_confidence,
            )
        return frame
