"""
utils.py
~~~~~~~~
Shared utility helpers for the ANP-Recognition pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_plate_annotation(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text: str,
    state: str = "",
    confidence: float = 0.0,
    box_colour: Tuple[int, int, int] = (0, 255, 0),
    text_colour: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a bounding box and label on *frame* (in-place).

    Parameters
    ----------
    frame:
        BGR image to annotate.
    x1, y1, x2, y2:
        Bounding box corners.
    text:
        Recognised plate text.
    state:
        Australian state abbreviation (optional).
    confidence:
        Detection confidence (0–1).
    box_colour, text_colour:
        BGR colour tuples.
    thickness:
        Line thickness.

    Returns
    -------
    The annotated frame (same array).
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, thickness)

    label_parts = []
    if text:
        label_parts.append(text)
    if state and state != "UNKNOWN":
        label_parts.append(f"[{state}]")
    if confidence > 0:
        label_parts.append(f"{confidence:.0%}")

    label = "  ".join(label_parts)
    if not label:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Background rectangle behind label.
    bg_x1, bg_y1 = x1, max(0, y1 - lh - baseline - 6)
    bg_x2, bg_y2 = x1 + lw + 8, y1
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), box_colour, cv2.FILLED)

    # White text on coloured background for contrast.
    cv2.putText(
        frame, label,
        (x1 + 4, max(lh, y1 - baseline - 2)),
        font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA,
    )
    return frame


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> Optional[np.ndarray]:
    """Load an image from *path* as a BGR NumPy array.

    Returns *None* and logs a warning when the file cannot be read.
    """
    img = cv2.imread(str(path))
    if img is None:
        logger.warning("Could not load image: %s", path)
    return img


def save_image(frame: np.ndarray, path: str | Path) -> bool:
    """Save *frame* to *path*.  Returns *True* on success."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), frame)
    if ok:
        logger.info("Saved annotated image: %s", out_path)
    else:
        logger.warning("Failed to save image: %s", out_path)
    return ok


def open_camera(source: int | str) -> cv2.VideoCapture:
    """Open a camera or video file.

    Parameters
    ----------
    source:
        Integer device index (e.g. ``0`` for the default webcam) or a path /
        URL string for a video file or RTSP stream.

    Returns
    -------
    An opened :class:`cv2.VideoCapture`.  Raises :class:`RuntimeError` if the
    source cannot be opened.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")
    logger.info(
        "Opened video source %r  (%dx%d @ %.1f fps)",
        source,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        cap.get(cv2.CAP_PROP_FPS),
    )
    return cap


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a simple format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )
