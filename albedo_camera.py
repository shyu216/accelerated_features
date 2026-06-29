#!/usr/bin/env python3
"""
Albedo live camera: classical Retinex or learned Careaga & Aksoy albedo.

With **no command-line arguments**, a **settings window** opens (tkinter).

  python albedo_camera.py

Use **no GUI** (automation / SSH):

  python albedo_camera.py --no-dialog
  python albedo_camera.py --no-dialog --method careaga --max-side 384 --pipeline-stage 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_repo_root = Path(__file__).resolve().parents[2]
_cam_depth = _repo_root / "camera" / "depth"
if str(_cam_depth) not in sys.path:
    sys.path.insert(0, str(_cam_depth))
from fps_meter import FpsMeter, draw_fps_bgr

from camera_pipeline import CameraConfig, CameraPipeline
from camera_pipeline.albedo import (
    dampen_chroma_bgr8,
    preview_log_shading_bgr8,
    retinex_gaussian_quotient_bgr8,
    retinex_luminance_log_bgr8,
    retinex_multiscale_bgr8,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Webcam albedo preview. Omit all args to open the settings window."
    )
    p.add_argument(
        "--no-dialog",
        action="store_true",
        help="Skip the tkinter settings panel; use flags below (or defaults).",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--buffer-size", type=int, default=1)
    p.add_argument("--disable-auto-exposure", action="store_true")
    p.add_argument("--auto-exposure", type=float, default=3.0)
    p.add_argument("--exposure", type=float, default=None)
    p.add_argument("--grabber-poll-sleep", type=float, default=0.0)
    p.add_argument(
        "--method",
        choices=("luminance", "per_channel", "gaussian_ratio", "careaga"),
        default="luminance",
    )
    p.add_argument("--sigma", type=float, default=25.0)
    p.add_argument(
        "--smooth-shading-sigma",
        type=float,
        default=14.0,
        help="Luminance: blur σ on log(s). 0=per-pixel Y(logI) (very detailed 'shading').",
    )
    p.add_argument("--careaga-version", default="v2")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-side", type=int, default=384)
    p.add_argument("--careaga-stride", type=int, default=2)
    p.add_argument(
        "--pipeline-stage",
        type=int,
        default=3,
        choices=(3, 4),
        help="Careaga: 3=albedo only (faster), 4=+diffuse/residual.",
    )
    p.add_argument(
        "--vivid",
        action="store_true",
        help="Classical: per-channel 2-98%% tone map (neon / high saturation).",
    )
    p.add_argument(
        "--view-gain",
        type=float,
        default=0.25,
        help="Luminance: 8-bit preview only, gain ∝ (this×255)/p99 luma (not in log Retinex eq).",
    )
    p.add_argument(
        "--max-display-scale",
        type=float,
        default=48.0,
        help="Luminance: max multiplier to 8-bit (safety only; try lower view-gain to dim).",
    )
    p.add_argument(
        "--chroma-damp",
        type=float,
        default=0.5,
        help="Classical: 0=grayscale, 1=unchanged. Reduces paint-like saturation in preview.",
    )
    p.add_argument(
        "--no-gray-world",
        action="store_true",
        help="Luminance: disable gray-world balance on the ratio (more color cast).",
    )
    p.add_argument(
        "--view-layout",
        choices=("albedo", "raw_albedo", "shading_albedo", "raw_shading_albedo"),
        default="shading_albedo",
        help="Preview: single albedo, raw|albedo, Y(logI) shading|albedo, or all three.",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Same as --view-layout raw_albedo (legacy).",
    )
    return p.parse_args()


def _albedo_classical(
    bgr: np.ndarray,
    method: str,
    sigma: float,
    smooth: float,
    vivid: bool = False,
    view_gain: float = 0.25,
    max_display_scale: float = 48.0,
    *,
    gray_world: bool = True,
    chroma_damp: float = 0.5,
) -> np.ndarray:
    if method == "luminance":
        return retinex_luminance_log_bgr8(
            bgr,
            smooth_shading_sigma=smooth,
            vivid=vivid,
            view_gain=view_gain,
            max_display_scale=max_display_scale,
            gray_world=gray_world,
            chroma_damp=chroma_damp,
        )
    if method == "per_channel":
        u8 = retinex_multiscale_bgr8(bgr, sigma=sigma, vivid=vivid)
        if not vivid and float(chroma_damp) < 0.999:
            u8 = dampen_chroma_bgr8(u8, float(chroma_damp))
        return u8
    if method == "gaussian_ratio":
        u8 = retinex_gaussian_quotient_bgr8(
            bgr, sigma=max(0.5, sigma * 0.5), vivid=vivid
        )
        if not vivid and float(chroma_damp) < 0.999:
            u8 = dampen_chroma_bgr8(u8, float(chroma_damp))
        return u8
    raise ValueError(method)


def run_albedo_session(
    *,
    method: str,
    width: int,
    height: int,
    cam: int,
    buffer_size: int,
    disable_auto_exposure: bool,
    auto_exposure: float,
    exposure: float | None,
    grabber_poll_sleep: float,
    sigma: float,
    smooth_shading_sigma: float,
    view_layout: str,
    careaga_version: str,
    device: str,
    max_side: int,
    careaga_stride: int,
    pipeline_stage: int,
    vivid: bool = False,
    view_gain: float = 0.25,
    max_display_scale: float = 48.0,
    gray_world: bool = True,
    chroma_damp: float = 0.5,
) -> int:
    careaga_m: Any = None
    last_careaga: np.ndarray | None = None
    fi = 0
    _careaga: Any = None

    if method == "careaga":
        try:
            from camera_pipeline.intrinsic_careaga import (
                careaga_hr_albedo_bgr,
                load_careaga_models,
            )
        except ImportError as e:  # pragma: no cover
            print(
                f"{e}\nInstall: pip install git+https://github.com/compphoto/Intrinsic",
                file=sys.stderr,
            )
            return 1
        _careaga = careaga_hr_albedo_bgr
        dev0 = None if device in ("", "auto") else device
        stg = int(max(3, min(4, int(pipeline_stage))))
        try:
            careaga_m = load_careaga_models(
                careaga_version, device=dev0, pipeline_stage=stg
            )
        except ImportError:
            print(
                "Careaga neural albedo requires the `intrinsic` package:\n"
                "  pip install git+https://github.com/compphoto/Intrinsic",
                file=sys.stderr,
            )
            return 1

    cfg = CameraConfig(
        device_index=cam,
        width=width,
        height=height,
        buffer_size=buffer_size,
        auto_exposure=None if disable_auto_exposure else auto_exposure,
        exposure=exposure,
        grabber_poll_sleep=grabber_poll_sleep,
    )
    camera = CameraPipeline(cfg)
    if not camera.open():
        print("Cannot open camera", file=sys.stderr)
        return 1
    camera.start()

    def _cap(img: np.ndarray, label: str) -> np.ndarray:
        o = img.copy()
        cv2.putText(
            o,
            label,
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (64, 255, 64),
            1,
            cv2.LINE_AA,
        )
        return o

    def _compose(frames: list[tuple[np.ndarray, str]]) -> np.ndarray:
        return np.hstack([_cap(f, lab) for f, lab in frames])

    layout = (view_layout or "shading_albedo").lower()
    if layout not in ("albedo", "raw_albedo", "shading_albedo", "raw_shading_albedo"):
        layout = "shading_albedo"
    n_show = {
        "albedo": 1,
        "raw_albedo": 2,
        "shading_albedo": 2,
        "raw_shading_albedo": 3,
    }[layout]

    win = "Albedo (q to quit) — " + layout.replace("_", " | ")
    if method == "careaga":
        win = f"Albedo — Careaga (q) — {layout.replace('_', ' | ')}"
    cv2.namedWindow(win, flags=cv2.WINDOW_GUI_NORMAL)
    wv = width * n_show
    cv2.resizeWindow(win, wv, height)
    fps_m = FpsMeter()
    try:
        while True:
            raw = camera.get_frame()
            if raw is None:
                continue
            if method == "careaga":
                fi += 1
                dev = None if device in ("", "auto") else device
                ms = max_side if max_side > 0 else None
                stg = int(max(3, min(4, int(pipeline_stage))))
                if last_careaga is None or (fi % max(1, careaga_stride)) == 0:
                    last_careaga = _careaga(
                        raw,
                        models=careaga_m,
                        version=careaga_version,
                        device=dev,
                        max_input_side=ms,
                        pipeline_stage=stg,
                    )
                alb = last_careaga
            else:
                alb = _albedo_classical(
                    raw,
                    method,
                    sigma,
                    smooth_shading_sigma,
                    vivid,
                    view_gain,
                    max_display_scale,
                    gray_world=gray_world,
                    chroma_damp=chroma_damp,
                )
            sh8 = preview_log_shading_bgr8(
                raw, smooth_shading_sigma=smooth_shading_sigma
            )
            aimg = alb
            if method == "careaga" and float(chroma_damp) < 0.999:
                aimg = dampen_chroma_bgr8(aimg, float(chroma_damp))
            if layout == "albedo":
                show = _cap(
                    aimg,
                    "albedo" if method != "careaga" else "Careaga albedo",
                )
            elif layout == "raw_albedo":
                show = _compose([(raw, "input"), (aimg, "albedo")])
            elif layout == "shading_albedo":
                show = _compose(
                    [
                        (sh8, "Y(log I)  shading"),
                        (aimg, "albedo"),
                    ]
                )
            else:
                show = _compose(
                    [
                        (raw, "input"),
                        (sh8, "Y(log I)  shading"),
                        (aimg, "albedo"),
                    ]
                )
            draw_fps_bgr(show, fps_m.tick(), anchor="tr")
            cv2.imshow(win, show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except ImportError as e:  # pragma: no cover
        print(e, file=sys.stderr)
        return 1
    except Exception as e:  # pragma: no cover
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        if method == "careaga" and "mps" in str(e).lower():
            print("Try: --device cpu", file=sys.stderr)
        return 1
    finally:
        camera.close()
        cv2.destroyAllWindows()
    return 0


def _settings_to_kwargs(s: Any) -> dict[str, Any]:
    """Map :class:`albedo_settings_ui.AlbedoAppSettings` to :func:`run_albedo_session`."""
    a = s
    return dict(
        method=a.method,
        width=a.width,
        height=a.height,
        cam=a.cam,
        buffer_size=a.buffer_size,
        disable_auto_exposure=a.disable_auto_exposure,
        auto_exposure=3.0,
        exposure=None,
        grabber_poll_sleep=0.0,
        sigma=a.sigma,
        smooth_shading_sigma=a.smooth_shading_sigma,
        view_layout=a.view_layout,
        careaga_version=a.careaga_version,
        device=a.device,
        max_side=a.max_side,
        careaga_stride=a.careaga_stride,
        pipeline_stage=a.pipeline_stage,
        vivid=a.vivid,
        view_gain=a.view_gain,
        max_display_scale=a.max_display_scale,
        gray_world=a.gray_world,
        chroma_damp=a.chroma_damp,
    )


def main() -> int:
    if "-h" in sys.argv or "--help" in sys.argv:
        _parse()
        return 0
    use_gui = len(sys.argv) <= 1

    if use_gui:
        try:
            from albedo_settings_ui import show_albedo_settings
        except ImportError as e:  # pragma: no cover
            print("Could not open settings UI: %s" % (e,), file=sys.stderr)
            return 1
        try:
            cfg = show_albedo_settings()
        except Exception as e:  # pragma: no cover
            print(
                "Tk error (%s). On Linux install python3-tk; use: python albedo_camera.py --no-dialog"
                % (e,),
                file=sys.stderr,
            )
            return 1
        if cfg is None:
            return 0
        return run_albedo_session(**_settings_to_kwargs(cfg))

    args = _parse()
    if not args.no_dialog:
        print("Pass --no-dialog when using command-line options.", file=sys.stderr)
        return 2
    vl = "raw_albedo" if args.compare else args.view_layout
    return run_albedo_session(
        method=args.method,
        width=args.width,
        height=args.height,
        cam=args.cam,
        buffer_size=args.buffer_size,
        disable_auto_exposure=args.disable_auto_exposure,
        auto_exposure=args.auto_exposure,
        exposure=args.exposure,
        grabber_poll_sleep=args.grabber_poll_sleep,
        sigma=args.sigma,
        smooth_shading_sigma=args.smooth_shading_sigma,
        view_layout=vl,
        careaga_version=args.careaga_version,
        device=args.device,
        max_side=args.max_side,
        careaga_stride=args.careaga_stride,
        pipeline_stage=args.pipeline_stage,
        vivid=args.vivid,
        view_gain=args.view_gain,
        max_display_scale=args.max_display_scale,
        gray_world=not args.no_gray_world,
        chroma_damp=args.chroma_damp,
    )


if __name__ == "__main__":
    raise SystemExit(main())
