"""Export the trained baseline to a CPU ONNX submission model and self-check it.

The exported graph mirrors the devkit baselines: single input ``windows``
``[N, 8, 100]`` and two outputs ``probabilities`` ``[N, 9]`` / ``relevance``
``[N, 8, 100]``, with a dynamic batch axis. The self-check reproduces the
devkit's own ``validate_submission`` gate (shape, finiteness, nonnegativity,
prob rows summing to 1, determinism) so packaging cannot fail on the interface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gearxai_workspace.data import NUM_CHANNELS, WINDOW_LENGTH
from gearxai_workspace.model import GearXAINet

INPUT_NAME = "windows"
OUTPUT_NAMES = ("probabilities", "relevance")
OPSET = 17


def export_onnx(
    model: GearXAINet,
    output_path: str | Path,
    *,
    opset: int = OPSET,
    sample: np.ndarray | None = None,
) -> Path:
    """Export ``model`` to ONNX and return the written path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    if sample is None:
        dummy = torch.randn(4, NUM_CHANNELS, WINDOW_LENGTH, dtype=torch.float32)
    else:
        dummy = torch.from_numpy(np.asarray(sample, dtype=np.float32))

    export_kwargs = dict(
        input_names=[INPUT_NAME],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={
            INPUT_NAME: {0: "N"},
            OUTPUT_NAMES[0]: {0: "N"},
            OUTPUT_NAMES[1]: {0: "N"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        # Legacy TorchScript exporter: avoids the onnxscript/dynamo dependency
        # and produces the small, static op graph the simplicity metric rewards.
        torch.onnx.export(model, dummy, str(output_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch without the ``dynamo`` keyword.
        torch.onnx.export(model, dummy, str(output_path), **export_kwargs)
    return output_path


def self_check(
    onnx_path: str | Path,
    sample_windows: np.ndarray,
    *,
    torch_model: GearXAINet | None = None,
) -> dict:
    """Validate the exported model and (optionally) compare to the torch outputs."""

    import onnxruntime as ort
    from gearxai_devkit.runtime import validate_submission

    onnx_path = Path(onnx_path)
    # Force a writable, contiguous copy (input may be a read-only memmap slice).
    sample = np.array(sample_windows[: min(8, len(sample_windows))], dtype=np.float32)

    # Devkit's own interface gate (raises on any violation).
    report = validate_submission(onnx_path, sample)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    probs, relevance = session.run(None, {INPUT_NAME: sample})

    result = {
        "devkit_validation": report,
        "prob_shape": list(probs.shape),
        "relevance_shape": list(relevance.shape),
        "prob_row_sum_range": [float(probs.sum(1).min()), float(probs.sum(1).max())],
        "relevance_min": float(relevance.min()),
    }

    if torch_model is not None:
        torch_model.eval()
        with torch.no_grad():
            t_probs, t_rel = torch_model(torch.from_numpy(sample))
        result["max_abs_prob_diff_vs_torch"] = float(np.abs(t_probs.numpy() - probs).max())
        result["max_abs_relevance_diff_vs_torch"] = float(np.abs(t_rel.numpy() - relevance).max())
    return result
