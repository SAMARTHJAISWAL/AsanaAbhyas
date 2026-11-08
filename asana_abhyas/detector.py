"""MediaPipe Tasks ``PoseLandmarker`` wrapper (replaces the removed ``mp.solutions.pose``)."""
from __future__ import annotations

import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{name}/float16/latest/pose_landmarker_{name}.task")
MODELS = ("lite", "full", "heavy")  # roughly model_complexity 0 / 1 / 2 in the old API


def model_cache_dir() -> Path:
    override = os.environ.get("ASANA_MODEL_DIR")
    return Path(override).expanduser() if override else Path.home() / ".cache" / "asana_abhyas"


def ensure_model(name: str = "full") -> Path:
    """Local path of a Pose Landmarker model, downloaded once into the cache dir."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODELS)}")
    target = model_cache_dir() / f"pose_landmarker_{name}.task"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model '{name}' to {target} ...")
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(MODEL_URL.format(name=name), tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target


class PoseDetector:
    """Detects one person's 33 landmarks in BGR images.

    ``video=False`` for unrelated still images (old ``static_image_mode=True``);
    ``video=True`` enables tracking between frames for webcam/video input.
    """

    def __init__(self, model: str = "full", model_path: Optional[str | Path] = None,
                 video: bool = False, min_confidence: float = 0.5) -> None:
        path = Path(model_path) if model_path else ensure_model(model)
        self._video = video
        self._last_ts = -1
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, image_bgr: np.ndarray):
        """Return ``(landmarks_px, world_landmarks)`` or ``(None, None)`` if nobody is found.

        ``landmarks_px`` is a list of 33 ``(x_px, y_px, z)`` tuples, as in the original notebook.
        """
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # must strictly increase
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)
        if not result.pose_landmarks:
            return None, None
        lms = result.pose_landmarks[0]
        points = [(lm.x * width, lm.y * height, lm.z * width) for lm in lms]
        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
        return points, world

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
