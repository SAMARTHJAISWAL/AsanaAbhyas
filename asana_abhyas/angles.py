"""Joint-angle geometry (pure Python)."""
from __future__ import annotations

import math
from typing import Sequence

Point = Sequence[float]


def calculate_angle(landmark1: Point, landmark2: Point, landmark3: Point) -> float:
    """Angle in degrees [0, 360) at ``landmark2`` going from ``landmark1`` to ``landmark3``.

    Only x and y are used. The angle is measured counter-clockwise in image coordinates, so it
    is not symmetric: swapping the outer points gives ``360 - angle``. The tree-pose rule
    relies on that to tell the left knee from the right.
    """
    x1, y1 = landmark1[0], landmark1[1]
    x2, y2 = landmark2[0], landmark2[1]
    x3, y3 = landmark3[0], landmark3[1]
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle
