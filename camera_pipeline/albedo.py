"""
Albedo *approximations* for real-time BGR input.

**Classical Retinex (luminance):** ``log(A) = log(I) - log(S)`` with
``log(S) = Y(log I)`` — luminance of the **log** B,G,R (ITU-R BT.601 weights
on B,G,R: 0.114, 0.587, 0.299). Equivalently ``A = I / exp(gray_log)`` (single
shading field). Optional Gaussian on ``log(S)`` (use a **non-zero** σ, e.g. 12–24)
so ``S`` is **low frequency**; with σ=0, ``Y(log I)`` is still *per pixel* and the
**shading** preview can look as detailed as a photograph (expected for local log).

**Multi-Scale Retinex (per-channel / ``per_channel`` mode):** three scales
(15, 80, 250 by default) of ``log I - log( Gaussian_σ I )`` averaged, ``cv2`` blur/log.
**Gaussian ratio:** per-channel ``R / (blur(R)+eps)``.

**Display:** the Retinex **estimate** is separate from 8-bit display. Default mapping uses
``view_gain`` and a luma-based reference; ``max_display_scale`` caps the multiplier as a
safety limit (raise it if the image stays dark). ``vivid=True`` use per-channel 2-98% histograms. 8-bit I is
not linear sRGB.
**Learned:** Careaga & Aksoy (``compphoto/Intrinsic``) — see
:mod:`camera_pipeline.intrinsic_careaga`.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from .config import FrameProcessor

AlbedoMode = Literal[
    "luminance",
    "per_channel",
    "gaussian_ratio",
    "retinex",
    "luminance_log",
    "msr",
    "old_retinex",
    "gaussian",
]

# Multi-Scale Retinex: fixed σ for Gaussian in linear domain (OpenCV (0,0), σ).
MSR_DEFAULT_SCALES: tuple[float, float, float] = (15.0, 80.0, 250.0)


def _bgr_luma(bgr: np.ndarray) -> np.ndarray:
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    return 0.114 * b + 0.587 * g + 0.299 * r


def _gray_world_balance_bgr(
    x: np.ndarray, *, eps: float = 1e-3, s_min: float = 0.35, s_max: float = 2.8
) -> np.ndarray:
    """
    Soft gray-world: scale B,G,R toward common mean. Gains are **clamped** so one
    under-filled channel (common after Retinex) cannot multiply by 30+ and break
    the global display tonemap (which used to make the albedo go black with noise).
    """
    t = np.asarray(x, dtype=np.float32)
    m = np.mean(t, axis=(0, 1))
    g = (float(m[0] + m[1] + m[2]) / 3.0) + eps
    s = g / (m + eps)
    s = np.clip(s, s_min, s_max)
    return np.clip(t * s.reshape(1, 1, 3), 0, None)


def dampen_chroma_bgr8(bgr: np.ndarray, factor: float) -> np.ndarray:
    """0 = grayscale, 1 = unchanged. OpenCV BGR L*a*b*."""
    f = float(np.clip(factor, 0.0, 1.0))
    if f >= 0.999:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b_ = lab[:, :, 2].astype(np.float32) - 128.0
    lab[:, :, 1] = np.clip(a * f + 128.0, 0, 255)
    lab[:, :, 2] = np.clip(b_ * f + 128.0, 0, 255)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def _percentile_normalize_gray_to_bgr8(s_log: np.ndarray) -> np.ndarray:
    """Map ``s_log`` to an 8-bit grayscale (visualize log-illumination / shading)."""
    sl = np.asarray(s_log, dtype=np.float32)
    lo, hi = np.percentile(sl, (1.0, 99.0))
    lo, hi = float(lo), float(hi)
    if hi < lo + 1e-6:
        t = np.full_like(sl, 64, dtype=np.uint8)
    else:
        t = (np.clip((sl - lo) / (hi - lo), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return np.dstack((t, t, t))


def log_shading_luma_bt601(
    bgr: np.ndarray,
    *,
    smooth_shading_sigma: float = 0.0,
) -> np.ndarray:
    """
    BT.601 weighted ``Y(log I)`` on 8-bit BGR (shared ``s_log`` with
    :func:`retinex_luminance_log_split`). For visualization, use
    :func:`log_shading_to_gray_bgr8` or a custom display.
    """
    f = np.clip(bgr.astype(np.float32), 1.0, 255.0)
    log_b, log_g, log_r = np.log(f[:, :, 0]), np.log(f[:, :, 1]), np.log(f[:, :, 2])
    s_log = 0.114 * log_b + 0.587 * log_g + 0.299 * log_r
    if smooth_shading_sigma and float(smooth_shading_sigma) > 0.0:
        s_log = cv2.GaussianBlur(
            s_log, (0, 0), float(smooth_shading_sigma)
        )
    return s_log


def log_shading_to_gray_bgr8(
    s_log: np.ndarray,
) -> np.ndarray:
    """``s_log`` (from :func:`log_shading_luma_bt601`) → 8-bit BGR gray preview."""
    return _percentile_normalize_gray_to_bgr8(s_log)


def preview_log_shading_bgr8(
    bgr: np.ndarray,
    *,
    smooth_shading_sigma: float = 0.0,
) -> np.ndarray:
    """
    ``Y(log I)`` (optionally ``GaussianBlur`` on that field) as 8-bit gray.

    If ``smooth_shading_sigma`` is **0**, ``Y(log I)`` is still per-pixel, so the
    preview can look as detailed as a photograph — that is expected for a **local**
    log-illuminant, not a global smooth light field. For a **low-frequency** illuminant
    in both the **equation and** this view, set the same non-zero
    :func:`log_shading_luma_bt601` / Retinex ``smooth_shading_sigma`` (e.g. 12–24).
    """
    s_log = log_shading_luma_bt601(
        bgr, smooth_shading_sigma=smooth_shading_sigma
    )
    return _percentile_normalize_gray_to_bgr8(s_log)


def _tonemap_luma_gain_bgr8(
    a_lin: np.ndarray,
    *,
    view_gain: float = 0.25,
    max_scale: float = 48.0,
) -> np.ndarray:
    """
    **Display only (not part of** ``log A = log I - log S``**):** one gain ``g`` on B,G,R:

    ``g = min( (view_gain·255) / y_ref(luma A),  max_scale )``

    ``y_ref`` blends the 99th luma with a fraction of the max (see code) so a tiny p99
    does not require absurd gain. ``max_scale`` is a **safety** bound; raise it in the UI
    if the preview is pegged. Lower ``view_gain`` gives a **darker** preview.
    """
    a_lin = np.clip(np.asarray(a_lin, dtype=np.float32), 0.0, None)
    y = _bgr_luma(a_lin)
    p99y = float(np.percentile(y, 99.0))
    p999y = float(np.percentile(y, 99.9))
    y_hi = max(p99y, 0.25 * p999y)
    y_hi = max(y_hi, 1e-3)
    vg = float(np.clip(view_gain, 0.05, 1.0))
    g = (vg * 255.0) / y_hi
    g = min(g, max(float(max_scale), 0.1))
    out = a_lin * g
    out = np.clip(out, 0.0, 255.0)
    if float(out.max()) < 2.0:
        p = max(float(np.percentile(a_lin, 99.5)), 1e-5)
        g2 = min(0.35 * 255.0 / p, max(float(max_scale), 0.1))
        out = np.clip(a_lin * g2, 0.0, 255.0)
    return (out + 0.5).astype(np.uint8)


def _stretch_rgb_channels_bgr8(
    f: np.ndarray,
    *,
    p_lo: float,
    p_hi: float,
) -> np.ndarray:
    out = np.empty_like(f, dtype=np.uint8)
    for c in range(3):
        plane = f[:, :, c]
        lo, hi = np.percentile(plane, (p_lo, p_hi))
        if hi - lo < 1e-6:
            out[:, :, c] = 128
        else:
            t = (plane - lo) / (hi - lo)
            out[:, :, c] = (np.clip(t, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return out


def retinex_luminance_log_split(
    bgr: np.ndarray,
    smooth_shading_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single-scale luminance Retinex in log domain: return ``(s_log, albedo_linear)`` with
    ``albedo_linear`` in linear RGB (still as BGR channel order).
    """
    f = np.clip(bgr.astype(np.float32), 1.0, 255.0)
    log_b = np.log(f[:, :, 0])
    log_g = np.log(f[:, :, 1])
    log_r = np.log(f[:, :, 2])
    s_log = 0.114 * log_b + 0.587 * log_g + 0.299 * log_r
    if smooth_shading_sigma and float(smooth_shading_sigma) > 0.0:
        s_log = cv2.GaussianBlur(s_log, (0, 0), float(smooth_shading_sigma))
    log_a = np.stack((log_b, log_g, log_r), axis=2) - s_log[:, :, None]
    albedo_lin = np.exp(np.clip(log_a, -30.0, 30.0))
    albedo_lin = np.clip(albedo_lin, 0.0, None)
    return s_log, albedo_lin


def retinex_luminance_log_bgr8(
    bgr: np.ndarray,
    *,
    smooth_shading_sigma: float = 0.0,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
    vivid: bool = False,
    view_gain: float = 0.25,
    max_display_scale: float = 48.0,
    gray_world: bool = True,
    chroma_damp: float = 0.5,
) -> np.ndarray:
    _, albedo_lin = retinex_luminance_log_split(
        bgr, smooth_shading_sigma
    )
    albedo_lin = np.clip(albedo_lin, 0.0, None)
    if gray_world:
        albedo_lin = _gray_world_balance_bgr(albedo_lin)
    if vivid:
        u8 = _stretch_rgb_channels_bgr8(albedo_lin, p_lo=p_lo, p_hi=p_hi)
    else:
        u8 = _tonemap_luma_gain_bgr8(
            albedo_lin, view_gain=view_gain, max_scale=max_display_scale
        )
    if chroma_damp < 0.999 and not vivid:
        u8 = dampen_chroma_bgr8(u8, chroma_damp)
    return u8


def multiscale_retinex_bgr_f32(
    bgr: np.ndarray,
    scales: tuple[float, ...] | None = None,
    eps: float = 1.0,
) -> np.ndarray:
    """
    **Multi-Scale Retinex (MSR)**, per channel, BGR input in 0–255:

    ``MSR = (1/K) sum_k  ( log(I+eps) - log( G_sigma_k(I) + eps) )``,

    with ``G_sigma = cv2.GaussianBlur(., (0,0), sigma)``. ``log`` is ``cv2.log``.

    Default ``scales`` is :data:`MSR_DEFAULT_SCALES` (15, 80, 250).
    """
    if scales is None or len(scales) == 0:
        scales = tuple(MSR_DEFAULT_SCALES)
    f = np.clip(bgr.astype(np.float32), 0.0, 255.0)
    h, w, _ = f.shape
    n = float(len(scales))
    out = np.empty_like(f, dtype=np.float32)
    for c in range(3):
        ch = f[:, :, c]
        acc = np.zeros((h, w), dtype=np.float32)
        for sig in scales:
            s = max(float(sig), 1e-3)
            blur = cv2.GaussianBlur(ch, (0, 0), s)
            a = (ch + eps).astype(np.float32)
            b = (np.maximum(blur, 0.0) + eps).astype(np.float32)
            log_a = np.empty((h, w), dtype=np.float32)
            log_b = np.empty((h, w), dtype=np.float32)
            cv2.log(a, log_a)
            cv2.log(b, log_b)
            acc = acc + (log_a - log_b)
        out[:, :, c] = acc / n
    return out


def multiscale_retinex_bgr8(
    bgr: np.ndarray,
    *,
    scales: tuple[float, ...] | None = None,
    eps: float = 1.0,
) -> np.ndarray:
    """
    :func:`multiscale_retinex_bgr_f32` then ``cv2.normalize`` to 0–255 per channel
    (same as the live ``msr_camera.py`` script).
    """
    m = multiscale_retinex_bgr_f32(bgr, scales=scales, eps=eps)
    h, w, _ = m.shape
    u8 = np.empty((h, w, 3), dtype=np.uint8)
    for c in range(3):
        u8[:, :, c] = cv2.normalize(
            m[:, :, c], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
    return u8


def retinex_multiscale_bgr8(
    bgr: np.ndarray,
    *,
    sigma: float = 25.0,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
    vivid: bool = False,
    scales: tuple[float, ...] | None = None,
    eps: float = 1.0,
) -> np.ndarray:
    """
    **Multi-Scale Retinex (MSR)**: :func:`multiscale_retinex_bgr_f32` with default
    three σ values (15, 80, 250). The ``sigma`` parameter is **unused** and kept
    for API compatibility with older callers; use ``scales=`` to override.

    - Default display: ``cv2.normalize`` to 0–255 **per channel** (same as ``msr_camera.py``).
    - ``vivid``: 2–98% per-channel stretch on the **float** MSR (garish / strong contrast).
    """
    _ = sigma
    m = multiscale_retinex_bgr_f32(bgr, scales=scales, eps=eps)
    if vivid:
        return _stretch_rgb_channels_bgr8(m, p_lo=p_lo, p_hi=p_hi)
    h, w, _ = m.shape
    u8 = np.empty((h, w, 3), dtype=np.uint8)
    for c in range(3):
        u8[:, :, c] = cv2.normalize(
            m[:, :, c], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
    return u8


def retinex_gaussian_quotient_bgr8(
    bgr: np.ndarray,
    *,
    sigma: float = 12.0,
    eps: float = 0.04,
    vivid: bool = False,
) -> np.ndarray:
    """Per-channel: R / (blur(R)+eps). Chroma-aware tonemap by default."""
    f = np.clip(bgr.astype(np.float32) / 255.0, 0.0, 1.0)
    out = np.empty_like(f)
    for c in range(3):
        ch = f[:, :, c]
        base = cv2.GaussianBlur(ch, (0, 0), float(sigma)) + float(eps)
        out[:, :, c] = ch / base
    out = np.clip(out, 0.0, None)
    m = out.max() + 1e-6
    a255 = (out * (1.0 / m)) * 255.0
    if vivid:
        return _stretch_rgb_channels_bgr8(
            a255, p_lo=2.0, p_hi=98.0
        )
    return _tonemap_luma_gain_bgr8(
        a255, view_gain=0.25, max_scale=48.0
    )


def make_albedo_frame_processor(
    mode: AlbedoMode = "luminance",
    *,
    sigma: float = 25.0,
    smooth_shading_sigma: float = 0.0,
    vivid: bool = False,
    view_gain: float = 0.25,
    max_display_scale: float = 48.0,
    gray_world: bool = True,
    chroma_damp: float = 0.5,
) -> FrameProcessor:
    m = (mode or "luminance").lower()
    if m in ("retinex", "luminance", "luminance_log"):
        def _f(bgr: np.ndarray) -> np.ndarray:
            return retinex_luminance_log_bgr8(
                bgr,
                smooth_shading_sigma=smooth_shading_sigma,
                vivid=vivid,
                view_gain=view_gain,
                max_display_scale=max_display_scale,
                gray_world=gray_world,
                chroma_damp=chroma_damp,
            )

        return _f
    if m in ("per_channel", "msr", "old_retinex"):
        def _f2(bgr: np.ndarray) -> np.ndarray:
            u8 = retinex_multiscale_bgr8(bgr, sigma=sigma, vivid=vivid)
            if not vivid and chroma_damp < 0.999:
                u8 = dampen_chroma_bgr8(u8, float(chroma_damp))
            return u8

        return _f2
    if m in ("gaussian", "gaussian_ratio"):
        def _f3(bgr: np.ndarray) -> np.ndarray:
            u8 = retinex_gaussian_quotient_bgr8(bgr, sigma=sigma, vivid=vivid)
            if not vivid and chroma_damp < 0.999:
                u8 = dampen_chroma_bgr8(u8, float(chroma_damp))
            return u8

        return _f3
    raise ValueError(f"Unknown albedo mode: {mode!r}")
