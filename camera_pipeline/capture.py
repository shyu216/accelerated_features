"""
Threaded frame acquisition and OpenCV capture setup.
"""

from __future__ import annotations

import threading
from time import sleep
from typing import Optional

import cv2
import numpy as np

from .config import CameraConfig, FrameProcessor


def _apply_processors(
    frame: np.ndarray, processors: list[FrameProcessor]
) -> np.ndarray:
    out = frame
    for fn in processors:
        out = fn(out)
    return out


def open_capture(config: CameraConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(config.device_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_FPS, config.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, config.buffer_size)

    if config.auto_exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, config.auto_exposure)
    if config.exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, config.exposure)
    return cap


class FrameGrabber(threading.Thread):
    """
    Continuously reads frames in the background. Latest frame is written under a lock
    so the consumer always sees a consistent image.
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        *,
        frame_processors: Optional[list[FrameProcessor]] = None,
        poll_sleep_s: float = 0.0,
    ):
        super().__init__(daemon=True)
        self._cap = cap
        self._processors = frame_processors or []
        self._poll_sleep_s = poll_sleep_s
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self.running = False
        # Prime first frame synchronously
        ret, frame = self._cap.read()
        if ret and frame is not None:
            self._frame = _apply_processors(frame, self._processors)

    def run(self) -> None:
        self.running = True
        while self.running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                sleep(0.01)
                continue
            processed = _apply_processors(frame, self._processors)
            with self._lock:
                self._frame = processed
            if self._poll_sleep_s > 0:
                sleep(self._poll_sleep_s)

    def stop(self) -> None:
        self.running = False

    def get_last_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()
