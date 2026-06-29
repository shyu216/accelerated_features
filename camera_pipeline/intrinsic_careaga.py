"""
High-quality albedo via **Careaga & Aksoy** (TOG 2023/2024), using the
public ``compphoto/Intrinsic`` package (*Colorful Diffuse Intrinsic Decomposition* pipeline).

This is *not* real-time on CPU for full VGA; use ``--max-side`` to cap input.

Paper: https://yaksoy.github.io/ColorfulShading/
Code:  https://github.com/compphoto/Intrinsic
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

import cv2
import numpy as np

_CACHED: dict[str, Any] = {}


def _pick_device(requested: Optional[str] = None) -> str:
    if requested and requested not in ("auto", ""):
        return requested
    t = importlib.import_module("torch")
    if t.cuda.is_available():
        return "cuda"
    if getattr(t.backends, "mps", None) and t.backends.mps.is_available():
        return "mps"
    return "cpu"


def _require_intrinsic() -> tuple[Any, Any]:
    try:
        from intrinsic.pipeline import load_models, run_pipeline
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Install the Intrinsic repo for neural albedo:\n"
            "  pip install git+https://github.com/compphoto/Intrinsic"
        ) from e
    return load_models, run_pipeline


def load_careaga_models(
    version: str = "v2",
    *,
    device: Optional[str] = "auto",
    pipeline_stage: int = 3,
) -> Any:
    """
    ``pipeline_stage`` 3: load up to the **albedo** net (skip diffuse ≈ 1 fewer stage file).
    4: full pipeline including diffuse / residual.
    """
    load_models, _ = _require_intrinsic()
    dev = _pick_device(device)
    st = int(np.clip(int(pipeline_stage), 1, 4))
    k = f"{version}:{dev}:{st}"
    if _CACHED.get("key") == k and _CACHED.get("models") is not None:
        return _CACHED["models"]
    m = load_models(version, device=dev, stage=st)
    _CACHED["key"] = k
    _CACHED["models"] = m
    return m


def careaga_hr_albedo_bgr(
    bgr: np.ndarray,
    *,
    models: Optional[Any] = None,
    version: str = "v2",
    device: Optional[str] = "auto",
    max_input_side: Optional[int] = 768,
    pipeline_stage: int = 3,
) -> np.ndarray:
    """
    Colorful / ordinal → **albedo** BGR. ``pipeline_stage`` 3 = albedo only (no diffuse net);
    4 = full (slower).
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    _, run_pipeline = _require_intrinsic()
    dev = _pick_device(device)
    st = int(np.clip(int(pipeline_stage), 1, 4))
    if models is None:
        models = load_careaga_models(version, device=dev, pipeline_stage=st)
    h0, w0 = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rh, rw = h0, w0
    if max_input_side and max(rh, rw) > max_input_side:
        s = max_input_side / max(rh, rw)
        rw = int(round(w0 * s))
        rh = int(round(h0 * s))
        rgb = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)

    import torch

    if dev == "mps" and not torch.backends.mps.is_built():
        dev = "cpu"
    with torch.no_grad():
        out = run_pipeline(models, rgb, device=dev, resize_conf=0.0, stage=st)
    hr = np.clip(np.asarray(out["hr_alb"]), 0.0, 1.0)
    if hr.shape[0] != h0 or hr.shape[1] != w0:
        hr = cv2.resize(hr, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
    bgr_out = cv2.cvtColor((hr * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr_out
