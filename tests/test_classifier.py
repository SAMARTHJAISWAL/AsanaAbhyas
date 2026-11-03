import math

import pytest

from asana_abhyas import (T_POSE, TREE, UNKNOWN, WARRIOR_II, JointAngles, PoseLandmark as P,
                          calculate_angle, classify_angles, classify_pose)


def test_angle_matches_notebook_example():
    # Same dummy values as the original notebook, which printed 166.26373169437744.
    assert calculate_angle((558, 326, 0), (642, 333, 0), (718, 321, 0)) == pytest.approx(166.26373169437744)


@pytest.mark.parametrize("p1,p3,expected", [((0, 0), (2, 0), 180.0), ((0, 0), (1, 1), 270.0),
                                            ((1, 1), (0, 0), 90.0)])
def test_angle_orientation(p1, p3, expected):
    assert calculate_angle(p1, (1, 0), p3) == pytest.approx(expected)


def test_angle_range_and_reversal():
    a = calculate_angle((0, 0, 0), (5, 3, 0), (9, 1, 0))
    b = calculate_angle((9, 1, 0), (5, 3, 0), (0, 0, 0))
    assert 0 <= a < 360
    assert a + b == pytest.approx(360)


def angles(**kw):
    base = dict(left_elbow=180, right_elbow=180, left_shoulder=95, right_shoulder=95,
                left_knee=180, right_knee=180)
    base.update(kw)
    return JointAngles(**base)


@pytest.mark.parametrize("kw,label", [
    ({}, T_POSE),
    ({"right_knee": 105}, WARRIOR_II),
    ({"left_knee": 105}, WARRIOR_II),
    ({"left_shoulder": 170, "right_shoulder": 170, "right_knee": 35}, TREE),
    ({"left_shoulder": 170, "right_shoulder": 170, "left_knee": 325}, TREE),
    ({"left_shoulder": 20, "right_shoulder": 20}, UNKNOWN),          # arms down
    ({"left_elbow": 120}, UNKNOWN),                                  # bent arm
    ({"right_shoulder": 111.8, "right_knee": 105}, UNKNOWN),         # just outside the 80-110 band
])
def test_rules(kw, label):
    assert classify_angles(angles(**kw)) == label


def _synthetic_t_pose():
    pts = [(0.0, 0.0, 0.0)] * 33
    pts = list(pts)
    def put(lm, x, y):
        pts[lm] = (float(x), float(y), 0.0)
    # image coordinates: y grows downward; person faces the camera.
    put(P.RIGHT_SHOULDER, 60, 100); put(P.RIGHT_ELBOW, 10, 100); put(P.RIGHT_WRIST, -40, 100)
    put(P.LEFT_SHOULDER, 100, 100); put(P.LEFT_ELBOW, 150, 100); put(P.LEFT_WRIST, 200, 100)
    put(P.RIGHT_HIP, 65, 200); put(P.RIGHT_KNEE, 65, 280); put(P.RIGHT_ANKLE, 65, 360)
    put(P.LEFT_HIP, 95, 200); put(P.LEFT_KNEE, 95, 280); put(P.LEFT_ANKLE, 95, 360)
    return pts


def test_classify_pose_from_landmarks():
    label, a = classify_pose(_synthetic_t_pose())
    assert label == T_POSE
    assert a.left_elbow == pytest.approx(180)
    assert 80 < a.left_shoulder < 110 and 80 < a.right_shoulder < 110


def test_landmark_indices():
    assert P.LEFT_SHOULDER == 11 and P.RIGHT_ANKLE == 28 and len(P) == 33
