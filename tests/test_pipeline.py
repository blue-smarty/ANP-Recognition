"""
test_pipeline.py
~~~~~~~~~~~~~~~~
Unit tests for the ANP-Recognition pipeline.

These tests exercise the state classifier and the plate-level pipeline logic
without requiring Hailo hardware, ONNX models, or a real camera.
They use synthetic (programmatically-created) plate images so the full test
suite can run on any CI machine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is importable.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.anpr.state_classifier import (
    AustralianState,
    AustralianStateClassifier,
    StateClassification,
)
from src.anpr.plate_detector import PlateDetector, DetectedPlate
from src.anpr.plate_recognizer import PlateRecognizer
from src.anpr.pipeline import ANPRPipeline, PlateResult
from src.anpr.qt_runtime import configure_qt_font_dir
from src.anpr.utils import draw_plate_annotation


# ===========================================================================
# Qt runtime helpers
# ===========================================================================

class TestQtRuntime:
    def test_configure_qt_font_dir_prefers_first_font_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("QT_QPA_FONTDIR", raising=False)
        monkeypatch.delenv("FONTCONFIG_PATH", raising=False)
        monkeypatch.delenv("FONTCONFIG_FILE", raising=False)

        bundled = tmp_path / "bundled-fonts"
        bundled.mkdir()
        (bundled / "DejaVuSans.ttf").write_bytes(b"font")

        fallback = tmp_path / "fallback-fonts"
        fallback.mkdir()
        (fallback / "SystemSans.ttf").write_bytes(b"font")

        chosen = configure_qt_font_dir([bundled, fallback])

        assert chosen == bundled
        assert Path(os.environ["QT_QPA_FONTDIR"]) == bundled

    def test_configure_qt_font_dir_preserves_existing_setting(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        configured = tmp_path / "already-configured"
        configured.mkdir()
        monkeypatch.setenv("QT_QPA_FONTDIR", str(configured))
        monkeypatch.delenv("FONTCONFIG_PATH", raising=False)
        monkeypatch.delenv("FONTCONFIG_FILE", raising=False)

        chosen = configure_qt_font_dir([tmp_path / "unused"])

        assert chosen == configured
        assert Path(os.environ["QT_QPA_FONTDIR"]) == configured

    def test_configure_qt_font_dir_respects_fontconfig(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("QT_QPA_FONTDIR", raising=False)
        monkeypatch.setenv("FONTCONFIG_PATH", str(tmp_path))

        bundled = tmp_path / "bundled-fonts"
        bundled.mkdir()
        (bundled / "DejaVuSans.ttf").write_bytes(b"font")

        chosen = configure_qt_font_dir([bundled])

        assert chosen is None
        assert os.environ.get("QT_QPA_FONTDIR") is None


# ===========================================================================
# AustralianStateClassifier
# ===========================================================================

class TestAustralianStateClassifier:
    """Tests for the text-pattern and colour-based state classifier."""

    def setup_method(self) -> None:
        self.clf = AustralianStateClassifier(use_visual_hints=False)

    # --- pattern-based tests -----------------------------------------------

    @pytest.mark.parametrize(
        "plate_text,expected_state",
        [
            # NSW – 3L 2D 1L
            ("ABC12D", AustralianState.NSW),
            ("ZZZ99Z", AustralianState.NSW),
            # VIC – 1D 2L 1D 2L
            ("1AB2CD", AustralianState.VIC),
            ("9XY3ZA", AustralianState.VIC),
            # QLD – older 3D 3L
            ("123ABC", AustralianState.QLD),
            ("456XYZ", AustralianState.QLD),
            # SA  – 3L 3D
            ("ABC123", AustralianState.SA),
            # WA  – 1D 3L 3D
            ("1ABC234", AustralianState.WA),
            ("9XYZ001", AustralianState.WA),
            # TAS – 2L 2D 2L
            ("AB12CD", AustralianState.TAS),
            # NT  – 2L 3D 1L
            ("AB123C", AustralianState.NT),
        ],
    )
    def test_pattern_classification(
        self, plate_text: str, expected_state: AustralianState
    ) -> None:
        result = self.clf.classify(plate_text)
        assert result.state == expected_state, (
            f"Plate '{plate_text}': expected {expected_state.value}, "
            f"got {result.state.value}  (scores={result.scores})"
        )

    def test_unknown_plate_returns_unknown(self) -> None:
        result = self.clf.classify("X")
        assert result.state == AustralianState.UNKNOWN

    def test_empty_plate_returns_unknown(self) -> None:
        result = self.clf.classify("")
        assert result.state == AustralianState.UNKNOWN

    def test_confidence_between_0_and_1(self) -> None:
        result = self.clf.classify("1AB2CD")
        assert 0.0 <= result.confidence <= 1.0

    def test_scores_dict_contains_all_states(self) -> None:
        result = self.clf.classify("ABC123")
        for state in AustralianState:
            if state != AustralianState.UNKNOWN:
                assert state.value in result.scores

    def test_hyphenated_input_normalised(self) -> None:
        """Hyphens and spaces should be stripped before matching."""
        result_clean = self.clf.classify("ABC12D")
        result_hyphen = self.clf.classify("ABC-12D")
        result_spaced = self.clf.classify("ABC 12D")
        assert result_clean.state == result_hyphen.state == result_spaced.state

    # --- visual hints ------------------------------------------------------

    def test_wa_yellow_background_boosts_score(self) -> None:
        clf_visual = AustralianStateClassifier(use_visual_hints=True)
        # Create a yellow-ish BGR plate crop.
        yellow_plate = np.full((40, 120, 3), fill_value=[0, 180, 220], dtype=np.uint8)
        # WA plate text + yellow background should give WA a higher score.
        result = clf_visual.classify("1ABC234", plate_crop=yellow_plate)
        assert result.state == AustralianState.WA

    # --- display name helper -----------------------------------------------

    @pytest.mark.parametrize(
        "state,expected_name",
        [
            (AustralianState.NSW, "New South Wales"),
            (AustralianState.VIC, "Victoria"),
            (AustralianState.QLD, "Queensland"),
            (AustralianState.SA,  "South Australia"),
            (AustralianState.WA,  "Western Australia"),
            (AustralianState.TAS, "Tasmania"),
            (AustralianState.ACT, "Australian Capital Territory"),
            (AustralianState.NT,  "Northern Territory"),
            (AustralianState.UNKNOWN, "Unknown"),
        ],
    )
    def test_state_display_name(
        self, state: AustralianState, expected_name: str
    ) -> None:
        name = self.clf.state_display_name(state)
        assert name == expected_name


# ===========================================================================
# PlateDetector  (no model – full-frame passthrough mode)
# ===========================================================================

class TestPlateDetector:
    """Tests for the detector when no model path is provided."""

    def setup_method(self) -> None:
        # Detector with no model → passthrough (whole frame = one plate).
        self.detector = PlateDetector()

    def test_detect_returns_list(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(frame)
        assert isinstance(results, list)

    def test_detect_passthrough_single_plate(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(frame)
        assert len(results) == 1

    def test_detected_plate_covers_full_frame(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        plate = self.detector.detect(frame)[0]
        assert plate.x1 == 0 and plate.y1 == 0
        assert plate.x2 == 640 and plate.y2 == 480

    def test_crop_attached(self) -> None:
        frame = np.ones((100, 200, 3), dtype=np.uint8) * 128
        plate = self.detector.detect(frame)[0]
        assert plate.crop is not None
        assert plate.crop.shape == (100, 200, 3)

    def test_detected_plate_area(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        plate = self.detector.detect(frame)[0]
        assert plate.area == 100 * 200

    def test_detected_plate_confidence(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        plate = self.detector.detect(frame)[0]
        assert plate.confidence == 1.0


# ===========================================================================
# PlateRecognizer  (Tesseract fallback with a synthetic image)
# ===========================================================================

class TestPlateRecognizer:
    """Tests for the recognizer using a synthetic white plate image."""

    def test_recognize_returns_string(self) -> None:
        rec = PlateRecognizer(backend="tesseract")
        plate = np.ones((50, 200, 3), dtype=np.uint8) * 255
        result = rec.recognize(plate)
        assert isinstance(result, str)

    def test_recognize_empty_crop_returns_empty(self) -> None:
        rec = PlateRecognizer(backend="tesseract")
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        result = rec.recognize(empty)
        assert result == ""

    def test_recognize_none_returns_empty(self) -> None:
        rec = PlateRecognizer(backend="tesseract")
        result = rec.recognize(None)
        assert result == ""


# ===========================================================================
# ANPRPipeline  (end-to-end without models)
# ===========================================================================

class TestANPRPipeline:
    """Smoke tests for the pipeline wired with no-model components."""

    def _make_pipeline(self) -> ANPRPipeline:
        detector = PlateDetector()
        recognizer = PlateRecognizer(backend="tesseract")
        classifier = AustralianStateClassifier()
        return ANPRPipeline(
            detector=detector,
            recognizer=recognizer,
            classifier=classifier,
            show_window=False,
            save_dir=None,
        )

    def test_process_image_returns_list(self) -> None:
        pipeline = self._make_pipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pipeline.process_image(frame, annotate=False)
        assert isinstance(results, list)

    def test_process_image_one_result_per_plate(self) -> None:
        pipeline = self._make_pipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pipeline.process_image(frame, annotate=False)
        # No model → detector returns one passthrough plate.
        assert len(results) == 1

    def test_result_has_expected_fields(self) -> None:
        pipeline = self._make_pipeline()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        results = pipeline.process_image(frame, annotate=False)
        r = results[0]
        assert isinstance(r.plate_text, str)
        assert isinstance(r.state, AustralianState)
        assert isinstance(r.state_name, str)
        assert 0.0 <= r.detection_confidence <= 1.0
        assert 0.0 <= r.state_confidence <= 1.0
        assert len(r.bbox) == 4

    def test_annotate_does_not_raise(self) -> None:
        pipeline = self._make_pipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pipeline.process_image(frame, annotate=False)
        annotated = pipeline._annotate_frame(frame.copy(), results)
        assert annotated.shape == frame.shape

    def test_process_missing_image_returns_empty(self, tmp_path: Path) -> None:
        pipeline = self._make_pipeline()
        missing = tmp_path / "nonexistent.jpg"
        results = pipeline.process_image(missing, annotate=False)
        assert results == []

    def test_run_stream_headless_does_not_call_waitkey(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipeline = self._make_pipeline()

        class _DummyCapture:
            def __init__(self) -> None:
                self._reads = 0

            def read(self):
                if self._reads == 0:
                    self._reads += 1
                    return True, np.zeros((32, 64, 3), dtype=np.uint8)
                return False, None

            def release(self) -> None:
                return None

        monkeypatch.setattr("src.anpr.pipeline.open_camera", lambda _source: _DummyCapture())

        def _fail_waitkey(_delay: int) -> int:
            raise AssertionError("cv2.waitKey should not be called when show_window is False")

        monkeypatch.setattr("src.anpr.pipeline.cv2.waitKey", _fail_waitkey)
        monkeypatch.setattr("src.anpr.pipeline.cv2.destroyAllWindows", lambda: None)

        pipeline.run_stream(source=0, max_frames=1, annotate=False)


# ===========================================================================
# draw_plate_annotation utility
# ===========================================================================

class TestDrawPlateAnnotation:
    def test_returns_same_array(self) -> None:
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        out = draw_plate_annotation(frame, 10, 10, 100, 50, text="ABC123", state="NSW")
        assert out is frame

    def test_no_error_on_empty_text(self) -> None:
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        draw_plate_annotation(frame, 10, 10, 100, 50, text="")

    def test_no_error_near_top_edge(self) -> None:
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        draw_plate_annotation(frame, 5, 2, 100, 50, text="XYZ99A", state="QLD")
