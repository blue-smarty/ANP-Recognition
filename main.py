#!/usr/bin/env python3
"""
main.py
~~~~~~~
CLI entry point for the Australian Number Plate Recognition (ANP-Recognition)
system.

Examples
--------
Process a single image::

    python main.py image photo.jpg

Process a directory of images::

    python main.py image /path/to/images/

Run on the default webcam::

    python main.py camera

Run on a specific camera index or video file::

    python main.py camera --source 1
    python main.py camera --source /path/to/video.mp4

Run on an RTSP stream::

    python main.py camera --source rtsp://user:pass@192.168.1.100/stream

All options::

    python main.py --help
    python main.py image --help
    python main.py camera --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``src`` is importable when this
# script is run directly from the repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.anpr.pipeline import ANPRPipeline
from src.anpr.utils import setup_logging

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_image(args: argparse.Namespace, pipeline: ANPRPipeline) -> None:
    """Process one or more still images."""
    path = Path(args.input)

    if not path.exists():
        logger.error("Input path does not exist: %s", path)
        sys.exit(1)

    # Collect image files.
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )

    if not files:
        logger.error("No image files found at: %s", path)
        sys.exit(1)

    logger.info("Processing %d image(s)…", len(files))
    for img_path in files:
        logger.info("  → %s", img_path.name)
        results = pipeline.process_image(img_path, annotate=True)
        if not results:
            logger.info("  No plates detected.")
        for r in results:
            print(f"  {img_path.name}: {r}")


def cmd_camera(args: argparse.Namespace, pipeline: ANPRPipeline) -> None:
    """Run on a live camera feed or video file."""
    # Determine source: integer index or path/URL string.
    source_raw = args.source
    try:
        source = int(source_raw)
    except ValueError:
        source = source_raw

    logger.info("Starting stream from source: %s", source)
    pipeline.run_stream(
        source=source,
        max_frames=args.max_frames,
        annotate=True,
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anpr",
        description=(
            "Australian Number Plate Recognition (ANP-Recognition)\n"
            "Detects and reads license plates from Australian states and territories\n"
            "using Hailo8 neural acceleration or an ONNX/Tesseract CPU fallback."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        metavar="PATH",
        help="Path to configuration YAML (default: config/config.yaml).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        metavar="DIR",
        help="Directory to save annotated output images.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open an OpenCV display window (useful for headless systems).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- image sub-command ---
    img_p = sub.add_parser("image", help="Process still image(s).")
    img_p.add_argument(
        "input",
        metavar="PATH",
        help="Path to an image file or a directory of images.",
    )

    # --- camera sub-command ---
    cam_p = sub.add_parser("camera", help="Process a live camera or video stream.")
    cam_p.add_argument(
        "--source",
        default="0",
        metavar="SOURCE",
        help=(
            "Camera index (e.g. 0), video file path, or RTSP/HTTP URL "
            "(default: 0)."
        ),
    )
    cam_p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Stop processing after N frames (default: run until quit).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Load pipeline.
    try:
        pipeline = ANPRPipeline.from_config(args.config)
    except FileNotFoundError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to initialise pipeline: %s", exc)
        sys.exit(1)

    # Apply CLI overrides.
    if args.no_window:
        pipeline._show_window = False
    if args.save_dir:
        from pathlib import Path as _Path

        pipeline._save_dir = _Path(args.save_dir)
        pipeline._save_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch to sub-command.
    if args.command == "image":
        cmd_image(args, pipeline)
    elif args.command == "camera":
        cmd_camera(args, pipeline)


if __name__ == "__main__":
    main()
