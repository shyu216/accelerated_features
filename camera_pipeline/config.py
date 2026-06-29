"""
Configuration for the webcam / capture path used by the real-time XFeat demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

# BGR uint8 in -> BGR uint8 out (e.g. future albedo / illuminant normalization).
FrameProcessor = Callable[[np.ndarray], np.ndarray]


@dataclass
class CameraConfig:
    """OpenCV capture settings. Extend here as the pipeline grows (e.g. exposure sweep)."""

    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: float = 30.0
    # Smaller buffer = lower latency, more "live" stream (some backends ignore this).
    buffer_size: int = 1
    # OpenCV: common values are 0.25 (auto) / 0.75 (manual) on Linux V4L2, or 3 on macOS AVFoundation; driver-dependent.
    auto_exposure: Optional[float] = 3.0
    # If set, passed to CAP_PROP_EXPOSURE (meaning depends on driver; often use with auto off).
    exposure: Optional[float] = None
    # Optional delay after each successful read in the grabber thread (0 = as fast as the driver allows).
    grabber_poll_sleep: float = 0.0
    # Optional per-frame chain applied before the matching loop (e.g. future albedo / color constancy).
    frame_processors: list[FrameProcessor] = field(default_factory=list)

    def with_processor(self, fn: FrameProcessor) -> "CameraConfig":
        """Return a new config with an extra frame processor (does not mutate self)."""
        return replace(self, frame_processors=[*self.frame_processors, fn])
