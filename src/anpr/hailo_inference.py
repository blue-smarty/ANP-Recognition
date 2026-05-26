"""
hailo_inference.py
~~~~~~~~~~~~~~~~~~
Thin wrapper around the Hailo8 runtime (hailo_platform).

When Hailo hardware or the hailo_platform package is unavailable the module
degrades gracefully to an ONNX-Runtime backend so the rest of the pipeline
can be developed and tested on any machine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import the Hailo runtime. If not available, fall back to ONNX.
# ---------------------------------------------------------------------------
try:
    from hailo_platform import (  # type: ignore[import]
        HEF,
        VDevice,
        FormatType,
        HailoSchedulingAlgorithm,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
    )
    _HAILO_AVAILABLE = True
    logger.info("hailo_platform detected – Hailo8 backend active.")
except ImportError:
    _HAILO_AVAILABLE = False
    logger.warning(
        "hailo_platform not found. Falling back to ONNX-Runtime backend. "
        "Install the HailoRT wheel from https://hailo.ai/developer-zone/ "
        "to enable Hailo8 acceleration."
    )

try:
    import onnxruntime as ort  # type: ignore[import]
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Hailo8 inference engine
# ---------------------------------------------------------------------------

class HailoInferenceEngine:
    """Run inference on a Hailo8 device.

    Parameters
    ----------
    hef_path:
        Path to the compiled .hef model file.
    input_name:
        Name of the model's input tensor.  If *None* the first input is used.
    output_name:
        Name of the model's output tensor.  If *None* all outputs are returned.
    """

    def __init__(
        self,
        hef_path: str | Path,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> None:
        if not _HAILO_AVAILABLE:
            raise RuntimeError(
                "hailo_platform is not installed. Use ONNXInferenceEngine instead."
            )

        self._hef_path = Path(hef_path)
        if not self._hef_path.exists():
            raise FileNotFoundError(f"HEF file not found: {self._hef_path}")

        self._input_name = input_name
        self._output_name = output_name

        # Load the HEF and configure the virtual device.
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._device = VDevice(params)

        hef = HEF(str(self._hef_path))
        self._network_group = self._device.configure(hef)[0]
        self._ng_params = self._network_group.create_params()

        # Resolve tensor names.
        input_infos = hef.get_input_vstream_infos()
        output_infos = hef.get_output_vstream_infos()
        self._resolved_input = (
            input_name if input_name else input_infos[0].name
        )
        self._resolved_outputs = (
            [output_name] if output_name
            else [o.name for o in output_infos]
        )

        logger.info(
            "Loaded Hailo model: %s  (input=%s, outputs=%s)",
            self._hef_path.name,
            self._resolved_input,
            self._resolved_outputs,
        )

    def infer(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on a single pre-processed frame.

        Parameters
        ----------
        frame:
            Pre-processed NumPy array matching the model's expected input shape.

        Returns
        -------
        dict mapping output tensor name → NumPy array.
        """
        input_vstream_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )
        output_vstream_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )

        with InferVStreams(
            self._network_group, input_vstream_params, output_vstream_params
        ) as infer_pipeline:
            with self._network_group.activate(self._ng_params):
                input_data = {self._resolved_input: np.expand_dims(frame, axis=0)}
                return infer_pipeline.infer(input_data)

    def __del__(self) -> None:
        if hasattr(self, "_device"):
            del self._device


# ---------------------------------------------------------------------------
# ONNX-Runtime fallback engine
# ---------------------------------------------------------------------------

class ONNXInferenceEngine:
    """ONNX-Runtime inference engine used when Hailo hardware is absent.

    Parameters
    ----------
    onnx_path:
        Path to the .onnx model file.
    input_name:
        Input tensor name override.  Uses the model's first input when *None*.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        input_name: Optional[str] = None,
    ) -> None:
        if not _ONNX_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is not installed. "
                "Run:  pip install onnxruntime"
            )

        self._onnx_path = Path(onnx_path)
        if not self._onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {self._onnx_path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            str(self._onnx_path), providers=providers
        )

        self._input_name = (
            input_name
            if input_name
            else self._session.get_inputs()[0].name
        )

        logger.info("Loaded ONNX model: %s", self._onnx_path.name)

    def infer(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference.

        Parameters
        ----------
        frame:
            Pre-processed NumPy array (C, H, W) or (H, W, C) depending on the
            model; callers are responsible for correct layout.

        Returns
        -------
        dict mapping output tensor name → NumPy array.
        """
        inp = np.expand_dims(frame, axis=0).astype(np.float32)
        output_names = [o.name for o in self._session.get_outputs()]
        results = self._session.run(output_names, {self._input_name: inp})
        return dict(zip(output_names, results))


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_inference_engine(
    backend: str,
    hef_path: Optional[str | Path] = None,
    onnx_path: Optional[str | Path] = None,
    input_name: Optional[str] = None,
) -> HailoInferenceEngine | ONNXInferenceEngine:
    """Return the appropriate inference engine based on *backend*.

    Parameters
    ----------
    backend:
        ``"hailo"`` or ``"onnx"``.
    hef_path:
        Required when *backend* is ``"hailo"``.
    onnx_path:
        Required when *backend* is ``"onnx"``.
    input_name:
        Optional input tensor name override.
    """
    if backend == "hailo":
        if not _HAILO_AVAILABLE:
            logger.warning(
                "Hailo backend requested but hailo_platform not installed. "
                "Falling back to ONNX backend."
            )
            if onnx_path is None:
                raise ValueError(
                    "Hailo backend unavailable and no onnx_path provided for fallback."
                )
            return ONNXInferenceEngine(onnx_path, input_name=input_name)
        if hef_path is None:
            raise ValueError("hef_path is required for Hailo backend.")
        return HailoInferenceEngine(hef_path, input_name=input_name)

    if backend == "onnx":
        if onnx_path is None:
            raise ValueError("onnx_path is required for ONNX backend.")
        return ONNXInferenceEngine(onnx_path, input_name=input_name)

    raise ValueError(f"Unknown backend '{backend}'. Choose 'hailo' or 'onnx'.")
