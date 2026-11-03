"""Rule-based yoga pose classification from joint angles (same thresholds as the notebook)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .angles import calculate_angle
from .landmarks import PoseLandmark as P

UNKNOWN = "Unknown Pose"
WARRIOR_II = "Warrior II Pose"
T_POSE = "T Pose"
TREE = "Tree Pose"


@dataclass(frozen=True)
class JointAngles:
    left_elbow: float
    right_elbow: float
    left_shoulder: float
    right_shoulder: float
    left_knee: float
    right_knee: float


def joint_angles(landmarks: Sequence[Sequence[float]]) -> JointAngles:
    """Compute the six angles the rules use from 33 ``(x, y, z)`` landmarks."""
    lm = landmarks
    return JointAngles(
        left_elbow=calculate_angle(lm[P.LEFT_SHOULDER], lm[P.LEFT_ELBOW], lm[P.LEFT_WRIST]),
        right_elbow=calculate_angle(lm[P.RIGHT_SHOULDER], lm[P.RIGHT_ELBOW], lm[P.RIGHT_WRIST]),
        left_shoulder=calculate_angle(lm[P.LEFT_ELBOW], lm[P.LEFT_SHOULDER], lm[P.LEFT_HIP]),
        right_shoulder=calculate_angle(lm[P.RIGHT_HIP], lm[P.RIGHT_SHOULDER], lm[P.RIGHT_ELBOW]),
        left_knee=calculate_angle(lm[P.LEFT_HIP], lm[P.LEFT_KNEE], lm[P.LEFT_ANKLE]),
        right_knee=calculate_angle(lm[P.RIGHT_HIP], lm[P.RIGHT_KNEE], lm[P.RIGHT_ANKLE]),
    )


def _between(value: float, low: float, high: float) -> bool:
    return low < value < high


def classify_angles(a: JointAngles) -> str:
    """Warrior II, T pose, tree pose or unknown, from joint angles."""
    label = UNKNOWN
    arms_straight = _between(a.left_elbow, 165, 195) and _between(a.right_elbow, 165, 195)
    shoulders_90 = _between(a.left_shoulder, 80, 110) and _between(a.right_shoulder, 80, 110)
    one_leg_straight = _between(a.left_knee, 165, 195) or _between(a.right_knee, 165, 195)

    if arms_straight and shoulders_90:
        if one_leg_straight and (_between(a.left_knee, 90, 120) or _between(a.right_knee, 90, 120)):
            label = WARRIOR_II
        if _between(a.left_knee, 160, 195) and _between(a.right_knee, 160, 195):
            label = T_POSE

    # Bent knee angle depends on which leg is bent (see calculate_angle's orientation).
    if one_leg_straight and (_between(a.left_knee, 315, 335) or _between(a.right_knee, 25, 45)):
        label = TREE
    return label


def classify_pose(landmarks: Sequence[Sequence[float]]) -> tuple[str, JointAngles]:
    angles = joint_angles(landmarks)
    return classify_angles(angles), angles
