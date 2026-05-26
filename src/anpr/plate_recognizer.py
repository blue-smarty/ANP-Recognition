"""
plate_recognizer.py
~~~~~~~~~~~~~~~~~~~
Extracts the text string from a cropped license plate image.

Two back-ends are supported:
* **Hailo8 / ONNX LPRNet** – fast neural OCR (preferred).
* **Tesseract** – fallback requiring only ``pytesseract`` + Tesseract binary.
"""

from __future__ import annotations

import logging
import re
import string
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# LPRNet character vocabulary (36 chars + blank token for CTC).
_LPRNET_CHARS = (
    list(string.digits)           # 0-9
    + list(string.ascii_uppercase) # A-Z
    + ["-"]                        # separator
    + ["_"]                        # blank / padding
)

# Regex to strip everything that is not a digit or upper-case letter or hyphen.
_CLEAN_RE = re.compile(r"[^A-Z0-9\-]")


class PlateRecognizer:
    """Read text from a cropped license plate image.

    Parameters
    ----------
    backend:
        ``"hailo"``, ``"onnx"``, or ``"tesseract"``.
    hef_path:
        LPRNet .hef model path (Hailo backend).
    onnx_path:
        LPRNet .onnx model path (ONNX backend).
    input_width, input_height:
        Model input resolution expected by LPRNet.
    tesseract_config:
        Tesseract PSM/OEM config string used by the fallback.
    """

    def __init__(
        self,
        backend: str = "tesseract",
        hef_path: Optional[str | Path] = None,
        onnx_path: Optional[str | Path] = None,
        input_width: int = 94,
        input_height: int = 24,
        tesseract_config: str = (
            "--psm 8 --oem 3 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ),
    ) -> None:
        self._input_w = input_width
        self._input_h = input_height
        self._tesseract_config = tesseract_config
        self._engine = None
        self._backend = backend

        if backend in ("hailo", "onnx") and (hef_path or onnx_path):
            try:
                from .hailo_inference import create_inference_engine

                self._engine = create_inference_engine(
                    backend=backend,
                    hef_path=hef_path,
                    onnx_path=onnx_path,
                )
            except Exception as exc:
                logger.warning(
                    "Could not load OCR model (%s). Falling back to Tesseract.",
                    exc,
                )
                self._backend = "tesseract"

        if self._backend == "tesseract":
            self._init_tesseract()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize(self, plate_crop: np.ndarray) -> str:
        """Return the recognised text from *plate_crop* (BGR).

        The returned string contains only digits, upper-case letters and
        hyphens.  Returns an empty string when recognition fails.
        """
        if plate_crop is None or plate_crop.size == 0:
            return ""

        if self._engine is not None:
            return self._recognize_neural(plate_crop)
        return self._recognize_tesseract(plate_crop)

    # ------------------------------------------------------------------
    # Neural OCR (LPRNet / Hailo8 / ONNX)
    # ------------------------------------------------------------------

    def _recognize_neural(self, crop: np.ndarray) -> str:
        pre = self._preprocess_lprnet(crop)
        raw = self._engine.infer(pre)
        output = next(iter(raw.values()))
        return self._decode_ctc(output)

    def _preprocess_lprnet(self, crop: np.ndarray) -> np.ndarray:
        """Resize, normalise and transpose a plate crop for LPRNet input."""
        resized = cv2.resize(crop, (self._input_w, self._input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0          # normalise to [-1, 1]
        chw = np.transpose(rgb, (2, 0, 1))   # HWC → CHW
        return chw

    def _decode_ctc(self, output: np.ndarray) -> str:
        """Greedy CTC decoding of LPRNet output logits.

        Parameters
        ----------
        output:
            Shape ``(1, T, C)`` or ``(T, C)`` where T is the time dimension
            and C the number of characters in the vocabulary.
        """
        probs = np.squeeze(output)  # → (T, C)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]

        # Greedy: take argmax at each timestep.
        indices: List[int] = np.argmax(probs, axis=-1).tolist()

        # Collapse repeated characters and remove blanks (last class).
        blank_idx = len(_LPRNET_CHARS) - 1
        chars: List[str] = []
        prev = None
        for idx in indices:
            if idx != blank_idx and idx != prev:
                if idx < len(_LPRNET_CHARS):
                    chars.append(_LPRNET_CHARS[idx])
            prev = idx

        return "".join(chars)

    # ------------------------------------------------------------------
    # Tesseract fallback
    # ------------------------------------------------------------------

    def _init_tesseract(self) -> None:
        try:
            import pytesseract  # type: ignore[import]

            self._pytesseract = pytesseract
        except ImportError:
            logger.warning(
                "pytesseract not installed. Text recognition disabled. "
                "Run:  pip install pytesseract"
            )
            self._pytesseract = None

    def _recognize_tesseract(self, crop: np.ndarray) -> str:
        if self._pytesseract is None:
            return ""

        # Upscale small plates to help Tesseract.
        h, w = crop.shape[:2]
        scale = max(1.0, 64.0 / h)
        if scale > 1.0:
            crop = cv2.resize(
                crop, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        # Convert to greyscale and apply adaptive threshold.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        _, thresh = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        try:
            from PIL import Image  # type: ignore[import]

            pil_img = Image.fromarray(thresh)
            raw = self._pytesseract.image_to_string(
                pil_img, config=self._tesseract_config
            )
        except Exception as exc:
            logger.debug("Tesseract error: %s", exc)
            return ""

        cleaned = _CLEAN_RE.sub("", raw.upper().strip())
        return cleaned
