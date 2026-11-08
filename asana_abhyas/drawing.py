"""OpenCV drawing helpers (replacement for the removed mp.solutions.drawing_utils)."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .classifier import UNKNOWN
from .landmarks import POSE_CONNECTIONS


def draw_landmarks(image: np.ndarray, landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    pts = [(int(x), int(y)) for x, y, *_ in landmarks]
    for a, b in POSE_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(image, p, 4, (0, 0, 255), -1, cv2.LINE_AA)
    return image


def draw_label(image: np.ndarray, label: str) -> np.ndarray:
    color = (0, 0, 255) if label == UNKNOWN else (0, 255, 0)
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    return image
