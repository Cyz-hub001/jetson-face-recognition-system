import os
import time
from datetime import datetime
from threading import Event, RLock, Thread
from typing import Callable, Optional

import cv2
import numpy as np


class VideoRecorder:
    def __init__(self, config: dict, logger=None):
        self.save_dir = config.get("save_dir", "recordings")
        self.duration = config.get("duration", 30)
        self.fps = config.get("fps", 20)
        self.codec = config.get("codec", "mp4v")
        self.extension = config.get("extension", ".mp4")
        self.enabled = config.get("enabled", True)
        self.logger = logger

        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self._lock = RLock()
        self._writer: Optional[cv2.VideoWriter] = None
        self._recording = False
        self._start_time: float = 0
        self._current_file: Optional[str] = None
        self._frame_count = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.duration + 5)
        self._cleanup_writer()

    def start_recording(self, frame_source: Callable[[], Optional[np.ndarray]]) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._recording:
                self._info("Recording already in progress, skipping")
                return
            self._recording = True
            self._stop_event.clear()

        self._thread = Thread(
            target=self._record_loop,
            args=(frame_source,),
            name="VideoRecorder",
            daemon=True
        )
        self._thread.start()

    def _record_loop(self, frame_source: Callable[[], Optional[np.ndarray]]) -> None:
        try:
            filepath = self._generate_filepath()
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            first_frame = frame_source()
            if first_frame is None:
                self._warning("No frame available for recording, aborting")
                return

            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))

            if not writer.isOpened():
                self._error(f"Failed to open video writer: {filepath}")
                return

            self._writer = writer
            self._current_file = filepath
            self._frame_count = 0
            self._start_time = time.time()
            self._info(f"Recording started: {filepath}, duration={self.duration}s")

            writer.write(first_frame)
            self._frame_count += 1

            while not self._stop_event.is_set():
                elapsed = time.time() - self._start_time
                if elapsed >= self.duration:
                    break

                frame = frame_source()
                if frame is not None:
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h))
                    writer.write(frame)
                    self._frame_count += 1

                self._stop_event.wait(1.0 / self.fps)

        except Exception as e:
            self._error(f"Recording error: {e}")
        finally:
            self._cleanup_writer()
            elapsed = time.time() - self._start_time if self._start_time else 0
            self._info(
                f"Recording stopped: file={self._current_file}, "
                f"frames={self._frame_count}, elapsed={elapsed:.1f}s"
            )
            with self._lock:
                self._recording = False

    def _cleanup_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _generate_filepath(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unknown_{timestamp}{self.extension}"
        return os.path.join(self.save_dir, filename)

    def _info(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)

    def _warning(self, message: str) -> None:
        if self.logger:
            self.logger.warning(message)

    def _error(self, message: str) -> None:
        if self.logger:
            self.logger.error(message)
