"""
Reusable camera capture and preprocessing for the XFeat real-time demo.

`CameraPipeline` is the main entry. Use `CameraConfig` to tune resolution, FPS,
buffering, and optional per-frame stages (e.g. future albedo / lighting steps).
"""

from .albedo import (
    MSR_DEFAULT_SCALES,
    make_albedo_frame_processor,
    multiscale_retinex_bgr8,
    multiscale_retinex_bgr_f32,
    retinex_gaussian_quotient_bgr8,
    retinex_luminance_log_bgr8,
    retinex_multiscale_bgr8,
)
from .config import CameraConfig, FrameProcessor
from .capture import FrameGrabber, open_capture
from .pipeline import CameraPipeline

__all__ = [
    "CameraConfig",
    "FrameProcessor",
    "FrameGrabber",
    "open_capture",
    "CameraPipeline",
    "make_albedo_frame_processor",
    "MSR_DEFAULT_SCALES",
    "multiscale_retinex_bgr_f32",
    "multiscale_retinex_bgr8",
    "retinex_gaussian_quotient_bgr8",
    "retinex_luminance_log_bgr8",
    "retinex_multiscale_bgr8",
]
