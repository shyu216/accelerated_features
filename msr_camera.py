#!/usr/bin/env python3
"""
Real-time Multi-Scale Retinex (MSR) on a webcam: original | enhanced.

Uses :func:`camera_pipeline.albedo.multiscale_retinex_bgr_f32` with three
fixed Gaussian scales (15, 80, 250), per-channel log-domain SSR averaged,
then :func:`cv2.normalize` per channel to 0–255 for display.

Run from this directory (or with ``PYTHONPATH`` including it):

  python msr_camera.py
  python msr_camera.py --cam 0 --width 640 --height 480
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_msr_dir = os.path.dirname(os.path.abspath(__file__))
if _msr_dir not in sys.path:
    sys.path.insert(0, _msr_dir)

_repo_root = Path(__file__).resolve().parents[2]
_cam_depth = _repo_root / "camera" / "depth"
if str(_cam_depth) not in sys.path:
    sys.path.insert(0, str(_cam_depth))
from fps_meter import FpsMeter, draw_fps_bgr

from camera_pipeline.albedo import (  # noqa: E402
    MSR_DEFAULT_SCALES,
    multiscale_retinex_bgr_f32,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cam", type=int, default=0, help="OpenCV device index (default: 0).")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--eps",
        type=float,
        default=1.0,
        help="Offset inside log (default 1.0) to avoid log(0).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse()
    cap = cv2.VideoCapture(int(args.cam))
    if not cap.isOpened():
        print(f"Cannot open camera index {args.cam}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(args.width)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(args.height)

    win = "MSR: original (left) | Retinex (right)  —  q to quit"
    cv2.namedWindow(win, flags=cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(win, w * 2, h)

    fps_m = FpsMeter()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            m = multiscale_retinex_bgr_f32(bgr, scales=MSR_DEFAULT_SCALES, eps=float(args.eps))
            h0, w0, _ = m.shape
            out8 = np.empty((h0, w0, 3), dtype=np.uint8)
            for c in range(3):
                out8[:, :, c] = cv2.normalize(
                    m[:, :, c],
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U,
                )
            show = np.hstack((bgr, out8))
            draw_fps_bgr(show, fps_m.tick(), anchor="tr")
            cv2.imshow(win, show)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
