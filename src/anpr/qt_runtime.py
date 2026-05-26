"""Qt runtime helpers used before importing OpenCV GUI modules."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
_DEFAULT_FONT_DIRS = (
    Path(__file__).resolve().parent / "fonts",
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/local/share/fonts"),
    Path("/usr/share/fonts"),
)


def _contains_fonts(font_dir: Path) -> bool:
    return font_dir.is_dir() and any(
        child.is_file() and child.suffix.lower() in _FONT_EXTENSIONS
        for child in font_dir.iterdir()
    )


def configure_qt_font_dir(
    candidate_dirs: Optional[Iterable[Path | str]] = None,
) -> Optional[Path]:
    """Set ``QT_QPA_FONTDIR`` when Qt fonts have been bundled alongside the app."""
    existing = os.environ.get("QT_QPA_FONTDIR")
    if existing:
        return Path(existing)

    if os.environ.get("FONTCONFIG_PATH") or os.environ.get("FONTCONFIG_FILE"):
        return None

    directories = candidate_dirs or _DEFAULT_FONT_DIRS
    for candidate in directories:
        font_dir = Path(candidate)
        if _contains_fonts(font_dir):
            os.environ["QT_QPA_FONTDIR"] = str(font_dir)
            return font_dir

    return None
