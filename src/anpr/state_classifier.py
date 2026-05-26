"""
state_classifier.py
~~~~~~~~~~~~~~~~~~~~
Identifies the Australian state or territory from a recognised plate text
string and/or the visual appearance of the plate crop.

Strategy
--------
1. **Pattern matching** (primary) – each jurisdiction uses a distinct
   alphanumeric format that can be matched with a regular expression.
2. **Visual colour analysis** (secondary) – background colour and text
   colour provide additional evidence when patterns are ambiguous.
3. **Confidence scoring** – each rule contributes a score; the state with
   the highest total score is returned.

Australian plate format reference
----------------------------------
State  | Example      | Notes
-------|--------------|------------------------------------------
NSW    | ABC-12D      | 3 letters, 2 digits, 1 letter (post-2009)
       | AB-12-CD     | Older format
VIC    | 1AB-2CD      | 1 digit, 2 letters, 1 digit, 2 letters
       | ABC-123      | Older / personalised
QLD    | 123-ABC      | 3 digits, 3 letters (older)
       | ABC-12A      | Newer (post-2013)
SA     | ABC-123      | 3 letters, 3 digits
WA     | 1ABC 234     | 1 digit, 3 letters, 3 digits
       | 1AB-234      | Variant
TAS    | AB-12-CD     | 2 letters, 2 digits, 2 letters
       | ABC-123      | Standard
ACT    | YAB-00A      | Letter, 2 letters, 2 digits, 1 letter
NT     | CA-12-3B     | 2 letters, 2 digits, 1 digit, 1 letter
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class AustralianState(str, Enum):
    """Australian states and territories."""

    NSW = "NSW"
    VIC = "VIC"
    QLD = "QLD"
    SA = "SA"
    WA = "WA"
    TAS = "TAS"
    ACT = "ACT"
    NT = "NT"
    UNKNOWN = "UNKNOWN"


@dataclass
class StateClassification:
    """Result of a state classification attempt."""

    state: AustralianState
    confidence: float
    # All candidate states with individual scores.
    scores: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Each entry: (state, compiled regex, base score)
# Patterns are tried in order; the first match grants the base score.
# Multiple patterns can match (e.g. a generic format) so all matches are
# accumulated and the highest total wins.
_L = "[A-Z]"
_D = "[0-9]"

_PATTERNS: List[Tuple[AustralianState, re.Pattern[str], float]] = [
    # ----- NSW -----
    # Current:   ABC-12D  (3L 2D 1L)
    (AustralianState.NSW, re.compile(rf"^{_L}{{3}}{_D}{{2}}{_L}$"), 0.85),
    # Older:     ABCD-12  (4L 2D) custom
    (AustralianState.NSW, re.compile(rf"^{_L}{{4}}{_D}{{2}}$"), 0.70),

    # ----- VIC -----
    # Current:   1AB-2CD  (1D 2L 1D 2L)
    (AustralianState.VIC, re.compile(rf"^{_D}{_L}{{2}}{_D}{_L}{{2}}$"), 0.90),
    # Older:     ABC-123
    (AustralianState.VIC, re.compile(rf"^{_L}{{3}}{_D}{{3}}$"), 0.50),

    # ----- QLD -----
    # Current:   ABC-12A  (3L 2D 1L)  – same format as NSW; colour resolves
    (AustralianState.QLD, re.compile(rf"^{_L}{{3}}{_D}{{2}}{_L}$"), 0.70),
    # Older:     123-ABC  (3D 3L)
    (AustralianState.QLD, re.compile(rf"^{_D}{{3}}{_L}{{3}}$"), 0.85),

    # ----- SA -----
    # Standard:  ABC-123  (3L 3D)
    (AustralianState.SA, re.compile(rf"^{_L}{{3}}{_D}{{3}}$"), 0.80),

    # ----- WA -----
    # Standard:  1ABC234 or 1ABC-234 (1D 3L 3D)
    (AustralianState.WA, re.compile(rf"^{_D}{_L}{{3}}{_D}{{3}}$"), 0.90),
    (AustralianState.WA, re.compile(rf"^{_D}{_L}{{2}}{_D}{{3}}$"), 0.65),

    # ----- TAS -----
    # Standard:  AB-12-CD  → stripped to AB12CD (2L 2D 2L)
    (AustralianState.TAS, re.compile(rf"^{_L}{{2}}{_D}{{2}}{_L}{{2}}$"), 0.80),
    # Also standard: ABC-123
    (AustralianState.TAS, re.compile(rf"^{_L}{{3}}{_D}{{3}}$"), 0.45),

    # ----- ACT -----
    # Standard:  YAB-00A  → stripped to YAB00A (1L 2L 2D 1L = 3L 2D 1L)
    # This overlaps NSW; rely on visual colour to differentiate.
    (AustralianState.ACT, re.compile(rf"^{_L}{{3}}{_D}{{2}}{_L}$"), 0.65),

    # ----- NT -----
    # Standard:  CA12-3B  → stripped to CA123B (2L 3D 1L)
    (AustralianState.NT, re.compile(rf"^{_L}{{2}}{_D}{{3}}{_L}$"), 0.80),
    # Also:      CA-12-3B → CA123B same as above
]


# ---------------------------------------------------------------------------
# Colour hints  (BGR mean of the plate background)
# ---------------------------------------------------------------------------

# Each entry: (state, (B_min, G_min, R_min), (B_max, G_max, R_max), score)
_COLOUR_HINTS: List[Tuple[AustralianState, np.ndarray, np.ndarray, float]] = [
    # NSW – white background with blue/green registration strip (white dominant)
    (AustralianState.NSW,  np.array([180, 180, 180]), np.array([255, 255, 255]), 0.20),
    # VIC – blue / white or yellow themes; most common white
    (AustralianState.VIC,  np.array([180, 180, 180]), np.array([255, 255, 255]), 0.15),
    # QLD – white background with maroon text
    (AustralianState.QLD,  np.array([180, 180, 180]), np.array([255, 255, 255]), 0.15),
    # SA – white background, black text
    (AustralianState.SA,   np.array([180, 180, 180]), np.array([255, 255, 255]), 0.15),
    # WA – gold/yellow background
    (AustralianState.WA,   np.array([0, 120, 160]),   np.array([80, 220, 255]), 0.30),
    # TAS – white background
    (AustralianState.TAS,  np.array([180, 180, 180]), np.array([255, 255, 255]), 0.15),
    # ACT – white/yellow background
    (AustralianState.ACT,  np.array([180, 180, 180]), np.array([255, 255, 255]), 0.15),
    # NT – white background, red text
    (AustralianState.NT,   np.array([180, 180, 180]), np.array([255, 255, 255]), 0.15),
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class AustralianStateClassifier:
    """Identify the Australian state or territory of a license plate.

    Parameters
    ----------
    use_visual_hints:
        When *True* (default) the plate crop's background colour is used as
        additional evidence alongside the text pattern match.
    """

    def __init__(self, use_visual_hints: bool = True) -> None:
        self._use_visual = use_visual_hints

    def classify(
        self,
        plate_text: str,
        plate_crop: Optional[np.ndarray] = None,
    ) -> StateClassification:
        """Classify the state/territory from *plate_text* (and optionally the
        *plate_crop* image).

        Parameters
        ----------
        plate_text:
            Normalised plate string (uppercase, hyphens/spaces removed before
            matching).
        plate_crop:
            BGR crop of the plate image.  Used for colour hints when
            *use_visual_hints* is *True*.

        Returns
        -------
        :class:`StateClassification`
        """
        normalised = self._normalise(plate_text)
        scores: Dict[str, float] = {s.value: 0.0 for s in AustralianState}

        # Pattern matching.
        for state, pattern, base_score in _PATTERNS:
            if pattern.fullmatch(normalised):
                scores[state.value] += base_score

        # Visual colour hints.
        if self._use_visual and plate_crop is not None and plate_crop.size > 0:
            colour_scores = self._score_colour(plate_crop)
            for state_val, cscore in colour_scores.items():
                scores[state_val] += cscore

        # Find winner.
        best_state_val = max(scores, key=lambda k: scores[k])
        best_score = scores[best_state_val]

        if best_score <= 0:
            return StateClassification(
                state=AustralianState.UNKNOWN,
                confidence=0.0,
                scores=scores,
            )

        # Normalise confidence to [0, 1] using a simple sigmoid-like scale.
        total = sum(scores.values()) or 1.0
        confidence = min(1.0, best_score / total * len(scores))

        return StateClassification(
            state=AustralianState(best_state_val),
            confidence=round(confidence, 3),
            scores=scores,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        """Strip hyphens, spaces and convert to upper case."""
        return re.sub(r"[\s\-]", "", text.upper())

    def _score_colour(self, crop: np.ndarray) -> Dict[str, float]:
        """Return colour-hint scores for each state."""
        result: Dict[str, float] = {}
        # Compute mean BGR of the central 60% of the plate.
        h, w = crop.shape[:2]
        cy1, cy2 = int(h * 0.2), int(h * 0.8)
        cx1, cx2 = int(w * 0.1), int(w * 0.9)
        roi = crop[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            return result
        mean_bgr = roi.mean(axis=(0, 1))

        for state, lo, hi, score in _COLOUR_HINTS:
            if np.all(mean_bgr >= lo) and np.all(mean_bgr <= hi):
                result[state.value] = result.get(state.value, 0.0) + score

        return result

    def state_display_name(self, state: AustralianState) -> str:
        """Return the full jurisdiction name."""
        _NAMES = {
            AustralianState.NSW: "New South Wales",
            AustralianState.VIC: "Victoria",
            AustralianState.QLD: "Queensland",
            AustralianState.SA:  "South Australia",
            AustralianState.WA:  "Western Australia",
            AustralianState.TAS: "Tasmania",
            AustralianState.ACT: "Australian Capital Territory",
            AustralianState.NT:  "Northern Territory",
            AustralianState.UNKNOWN: "Unknown",
        }
        return _NAMES.get(state, state.value)
