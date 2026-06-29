"""
High-level camera source for the real-time demo: open, configure, stream latest frame.
"""

from __future__ import annotations

import cv2

from .capture import FrameGrabber, open_capture
from .config import CameraConfig


class CameraPipeline:
    """
    Owns VideoCapture + background grabber. Call start() then poll get_frame().
    Stages from config.frame_processors run inside the capture thread (before XFeat).
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self._cap: cv2.VideoCapture | None = None
        self._grabber: FrameGrabber | None = None

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def height(self) -> int:
        return self.config.height

    def open(self) -> bool:
        self._cap = open_capture(self.config)
        if not self._cap.isOpened():
            return False
        self._grabber = FrameGrabber(
            self._cap,
            frame_processors=self.config.frame_processors,
            poll_sleep_s=self.config.grabber_poll_sleep,
        )
        return True

    def start(self) -> None:
        if self._grabber is None:
            raise RuntimeError("Call open() before start()")
        self._grabber.start()

    def get_frame(self):
        if self._grabber is None:
            return None
        return self._grabber.get_last_frame()

    def close(self) -> None:
        if self._grabber is not None:
            self._grabber.stop()
            self._grabber.join(timeout=2.0)
            self._grabber = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
