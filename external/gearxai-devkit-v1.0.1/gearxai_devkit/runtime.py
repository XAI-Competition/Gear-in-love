"""ONNX Runtime validation and inference for GearXAI submissions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from gearxai_devkit.constants import NUM_CHANNELS, NUM_CLASSES, WINDOW_LENGTH
from gearxai_devkit.data import ensure_channels_first, iter_batches


@dataclass(frozen=True)
class RuntimeResult:
    probabilities: np.ndarray
    relevance: np.ndarray
    load_seconds: float
    inference_seconds: float
    input_name: str
    output_names: tuple[str, str]


def cpu_session(model_path: str | Path) -> tuple[ort.InferenceSession, float]:
    start = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    return session, time.perf_counter() - start


def looks_like_probabilities(array: np.ndarray) -> bool:
    return array.ndim == 2 and array.shape[1] == NUM_CLASSES


def looks_like_relevance(array: np.ndarray) -> bool:
    return array.ndim == 3 and array.shape[1:] == (NUM_CHANNELS, WINDOW_LENGTH)


def split_outputs(outputs: list[np.ndarray], output_names: list[str]) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    prob_idx = None
    rel_idx = None
    for idx, value in enumerate(outputs):
        if prob_idx is None and looks_like_probabilities(value):
            prob_idx = idx
        elif rel_idx is None and looks_like_relevance(value):
            rel_idx = idx

    if prob_idx is None or rel_idx is None:
        shapes = [tuple(v.shape) for v in outputs]
        raise ValueError(
            "Submission must return probabilities [N, 9] and relevance [N, 8, 100]. "
            f"Observed output shapes: {shapes}."
        )

    return outputs[prob_idx], outputs[rel_idx], (output_names[prob_idx], output_names[rel_idx])


def run_submission(
    model_path: str | Path,
    windows: np.ndarray,
    *,
    batch_size: int = 256,
) -> RuntimeResult:
    """Run a GearXAI ONNX submission on CPU."""

    windows = ensure_channels_first(windows)
    session, load_seconds = cpu_session(model_path)
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"Expected exactly one ONNX input, got {len(inputs)}.")
    input_name = inputs[0].name
    output_names = [out.name for out in session.get_outputs()]

    prob_chunks: list[np.ndarray] = []
    rel_chunks: list[np.ndarray] = []
    selected_names: tuple[str, str] | None = None
    start = time.perf_counter()
    for batch in iter_batches(windows, batch_size):
        outputs = session.run(None, {input_name: batch.astype(np.float32, copy=False)})
        probs, relevance, names = split_outputs(outputs, output_names)
        prob_chunks.append(np.asarray(probs, dtype=np.float32))
        rel_chunks.append(np.asarray(relevance, dtype=np.float32))
        selected_names = names
    inference_seconds = time.perf_counter() - start

    return RuntimeResult(
        probabilities=np.concatenate(prob_chunks, axis=0),
        relevance=np.concatenate(rel_chunks, axis=0),
        load_seconds=load_seconds,
        inference_seconds=inference_seconds,
        input_name=input_name,
        output_names=selected_names or ("", ""),
    )


def validate_submission(
    model_path: str | Path,
    sample_windows: np.ndarray,
    *,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Validate signature, finiteness, nonnegative relevance, and determinism."""

    model = onnx.load(str(model_path))
    quantized_ops = {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul"}
    present_quantized_ops = sorted({node.op_type for node in model.graph.node} & quantized_ops)
    if present_quantized_ops:
        raise ValueError(f"Quantized ONNX operators are not allowed: {present_quantized_ops}.")

    sample_windows = ensure_channels_first(sample_windows)
    first = run_submission(model_path, sample_windows)
    second = run_submission(model_path, sample_windows)

    probs = first.probabilities
    relevance = first.relevance
    if probs.shape != (len(sample_windows), NUM_CLASSES):
        raise ValueError(f"Invalid probability shape: {probs.shape}.")
    if relevance.shape != (len(sample_windows), NUM_CHANNELS, WINDOW_LENGTH):
        raise ValueError(f"Invalid relevance shape: {relevance.shape}.")
    if not np.all(np.isfinite(probs)):
        raise ValueError("Probability output contains NaN or Inf values.")
    if np.min(probs) < -atol:
        raise ValueError("Probability output must be nonnegative.")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3, rtol=1e-3):
        raise ValueError("Probability rows must sum to 1.")
    if not np.all(np.isfinite(relevance)):
        raise ValueError("Relevance output contains NaN or Inf values.")
    if np.min(relevance) < -atol:
        raise ValueError("Relevance output must be nonnegative.")
    if not np.allclose(probs, second.probabilities, atol=atol, rtol=atol):
        raise ValueError("Probability output is not deterministic across repeated runs.")
    if not np.allclose(relevance, second.relevance, atol=atol, rtol=atol):
        raise ValueError("Relevance output is not deterministic across repeated runs.")

    return {
        "valid": True,
        "samples": int(len(sample_windows)),
        "input_name": first.input_name,
        "probability_output": first.output_names[0],
        "relevance_output": first.output_names[1],
        "load_seconds": first.load_seconds,
        "inference_seconds": first.inference_seconds,
        "probability_sum_min": float(row_sums.min()),
        "probability_sum_max": float(row_sums.max()),
        "relevance_min": float(relevance.min()),
        "relevance_max": float(relevance.max()),
    }
