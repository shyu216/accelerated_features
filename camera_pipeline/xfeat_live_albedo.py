"""
Optional **live** BGR albedo for XFeat + localization, without changing reference images.

`process/localization_testing` scripts can add the CLI group from
:func:`add_live_albedo_argparse` and, after each ``read()`` + resize, run::

    albedo_fn = live_albedo_from_args(args)  # once at startup, if albedo
    if albedo_fn is not None:
        frame = albedo_fn(frame)

References / orbit pano Renders stay as plain RGB; only the current camera frame
is processed. Classical modes use :func:`albedo.make_albedo_frame_processor`.
**careaga** uses :mod:`intrinsic_careaga` (Careaga & Aksoy; requires ``compphoto/Intrinsic``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .albedo import dampen_chroma_bgr8, make_albedo_frame_processor


def add_live_albedo_argparse(
    parser: Any,
) -> Any:
    """
    Add ``--albedo`` and tuning flags. Use ``nargs="?"`` so
    ``--albedo`` alone enables default ``luminance``, or pass e.g. ``--albedo per_channel``.
    """
    g = parser.add_argument_group(
        "Live albedo (optional)",
        "Process the live BGR stream only; references are unchanged. "
        "Classical: see camera_pipeline/albedo.py. "
        "careaga: Careaga & Aksoy (pip install git+https://github.com/compphoto/Intrinsic).",
    )
    g.add_argument(
        "--albedo",
        nargs="?",
        const="luminance",
        default=None,
        choices=["luminance", "per_channel", "gaussian_ratio", "careaga"],
        metavar="MODE",
        help="Enable albedo: luminance (default if --albedo alone), per_channel (MSR), "
        "gaussian_ratio, or careaga (learned; use --albedo-careaga-* for speed).",
    )
    g.add_argument(
        "--albedo-smooth",
        type=float,
        default=14.0,
        help="Luminance mode: σ for Gaussian on log(s). (Ignored for MSR / gaussian.)",
    )
    g.add_argument(
        "--albedo-sigma",
        type=float,
        default=12.0,
        help="gaussian_ratio: blur σ. per_channel/MSR ignores this (fixed scales).",
    )
    g.add_argument(
        "--albedo-vivid",
        action="store_true",
        help="Classical: garish 2-98%% per-channel tone map (not for luminance ratio display).",
    )
    g.add_argument(
        "--albedo-view-gain",
        type=float,
        default=0.25,
        help="Luminance: 8-bit preview gain (live frame only).",
    )
    g.add_argument(
        "--albedo-max-scale",
        type=float,
        default=48.0,
        help="Luminance: max display scale cap.",
    )
    g.add_argument(
        "--albedo-chroma",
        type=float,
        default=0.5,
        help="0=gray … 1=raw chroma (luminance: after ratio; all modes: on 8-bit preview).",
    )
    g.add_argument(
        "--albedo-no-gray-world",
        action="store_true",
        help="Luminance: disable gray-world on the ratio.",
    )
    g.add_argument(
        "--albedo-input-stride",
        type=int,
        default=1,
        metavar="N",
        help="Recompute albedo only every N camera frames (reuse previous for XFeat+viz). "
        "E.g. N=3 at ~30 fps ≈10 effective albedo updates/s for any mode. Default 1=every frame.",
    )
    g.add_argument(
        "--albedo-careaga-version",
        type=str,
        default="v2",
        help="careaga: Intrinsic weight pack (e.g. v2, v2.1).",
    )
    g.add_argument(
        "--albedo-careaga-max-side",
        type=int,
        default=256,
        help="careaga: max input side in px (lower=faster; default 256 = fast preset).",
    )
    g.add_argument(
        "--albedo-careaga-stride",
        type=int,
        default=10,
        help="careaga: run the network every N live frames (default 10); reuse in between. "
        "Combine with --albedo-input-stride only if you need an extra global throttle.",
    )
    g.add_argument(
        "--albedo-careaga-pipeline-stage",
        type=int,
        default=3,
        choices=(3, 4),
        help="careaga: 3=albedo only (faster), 4=full pipeline.",
    )
    g.add_argument(
        "--albedo-careaga-device",
        type=str,
        default="auto",
        help="careaga: auto | cpu | cuda | mps",
    )
    return g


def live_albedo_from_args(args: Any) -> Callable[[np.ndarray], np.ndarray] | None:
    """
    Build a single callable ``(bgr uint8) -> bgr uint8`` from a parsed namespace
    that includes :func:`add_live_albedo_argparse` fields, or return ``None``.
    """
    m = getattr(args, "albedo", None)
    if m is None or (isinstance(m, str) and m.lower() in ("", "off", "none")):
        return None
    mode = str(m).lower()
    if mode == "careaga":
        fn = _live_albedo_careaga_from_args(args)
    else:
        ap = {
            "mode": mode,
            "sigma": float(getattr(args, "albedo_sigma", 12.0)),
            "smooth_shading_sigma": float(getattr(args, "albedo_smooth", 14.0)),
            "vivid": bool(getattr(args, "albedo_vivid", False)),
            "view_gain": float(getattr(args, "albedo_view_gain", 0.25)),
            "max_display_scale": float(getattr(args, "albedo_max_scale", 48.0)),
            "gray_world": not bool(getattr(args, "albedo_no_gray_world", False)),
            "chroma_damp": float(getattr(args, "albedo_chroma", 0.5)),
        }
        fn = make_albedo_frame_processor(
            ap["mode"],
            sigma=ap["sigma"],
            smooth_shading_sigma=ap["smooth_shading_sigma"],
            vivid=ap["vivid"],
            view_gain=ap["view_gain"],
            max_display_scale=ap["max_display_scale"],
            gray_world=ap["gray_world"],
            chroma_damp=ap["chroma_damp"],
        )
    return _wrap_albedo_input_stride(fn, args)


def _wrap_albedo_input_stride(
    fn: Callable[[np.ndarray], np.ndarray],
    args: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    """Only call ``fn`` every N camera frames; forward the last BGR in between (any albedo mode)."""
    n = max(1, int(getattr(args, "albedo_input_stride", 1)))
    if n <= 1:
        return fn
    last: list[np.ndarray | None] = [None]
    ct = [0]

    def _wrapped(bgr: np.ndarray) -> np.ndarray:
        ct[0] += 1
        if last[0] is None or (ct[0] % n == 0):
            last[0] = fn(bgr)
        return last[0] if last[0] is not None else bgr

    return _wrapped


def _live_albedo_careaga_from_args(
    args: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    from .intrinsic_careaga import careaga_hr_albedo_bgr, load_careaga_models

    version = str(getattr(args, "albedo_careaga_version", "v2") or "v2").strip()
    max_side = int(getattr(args, "albedo_careaga_max_side", 256) or 0)
    if max_side <= 0:
        max_side = None
    stride = max(1, int(getattr(args, "albedo_careaga_stride", 10)))
    stg = int(max(3, min(4, int(getattr(args, "albedo_careaga_pipeline_stage", 3)))))
    dev: str | None = getattr(args, "albedo_careaga_device", "auto") or "auto"
    if dev in ("", "auto"):
        dev = None
    chroma = float(getattr(args, "albedo_chroma", 0.5))

    models = load_careaga_models(version, device=dev, pipeline_stage=stg)
    last: list[np.ndarray | None] = [None]
    fi = [0]

    def _fn(bgr: np.ndarray) -> np.ndarray:
        fi[0] += 1
        if last[0] is None or (fi[0] % stride) == 0:
            last[0] = careaga_hr_albedo_bgr(
                bgr,
                models=models,
                version=version,
                device=dev,
                max_input_side=max_side,
                pipeline_stage=stg,
            )
        out = last[0]
        if out is None:  # pragma: no cover
            return bgr
        if chroma < 0.999:
            out = dampen_chroma_bgr8(out, chroma)
        return out

    return _fn


def describe_live_albedo_from_args(args: Any) -> str:
    """One-line for logging when albedo is on."""
    m = getattr(args, "albedo", None)
    if m is None:
        return "off"
    ins = int(getattr(args, "albedo_input_stride", 1))
    ins_s = f"  input_stride={ins}" if ins > 1 else ""
    if str(m).lower() == "careaga":
        return (
            f"careaga  max_side={getattr(args, 'albedo_careaga_max_side', 256)}  "
            f"net_stride={getattr(args, 'albedo_careaga_stride', 10)}  "
            f"stage={getattr(args, 'albedo_careaga_pipeline_stage', 3)}  "
            f"device={getattr(args, 'albedo_careaga_device', 'auto')!r}  "
            f"chroma={getattr(args, 'albedo_chroma', 0.5):g}{ins_s}"
        )
    return (
        f"{m}  (smooth={getattr(args, 'albedo_smooth', 14.0):g}, "
        f"chroma={getattr(args, 'albedo_chroma', 0.5):g}, "
        f"vivid={getattr(args, 'albedo_vivid', False)}){ins_s}"
    )
