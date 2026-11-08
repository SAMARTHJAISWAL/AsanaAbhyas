# AsanaAbhyas

Yoga pose detection and classification with MediaPipe. A person's 33 body landmarks are
detected with the MediaPipe **Pose Landmarker**. Joint angles at the elbows, shoulders and knees
are computed from them, and simple angle rules recognise three poses: **Warrior II**, **T Pose**
and **Tree Pose**. It works on photos, video files, and a live webcam.

The project has two parts: a tutorial-style notebook (`AsanaAbhyas.ipynb`) and the same logic as
a small Python package with a command-line tool (`asana_abhyas/`).

## Features

- Landmark detection with MediaPipe Tasks `PoseLandmarker`. There are three model sizes:
  `lite`, `full` and `heavy`. The model is downloaded once to `~/.cache/asana_abhyas/`, or to
  `$ASANA_MODEL_DIR` if set.
- Joint-angle computation and rule-based classification (thresholds below).
- Annotated output: skeleton and pose label, saved as an image or video.
- Webcam mode with mirrored view and an FPS counter.
- Unit tests for the geometry and classification rules. They run without MediaPipe or a camera.

## Tech stack

Python 3.10+ (tested with 3.12), MediaPipe (Tasks API), OpenCV, NumPy, Matplotlib, Jupyter, pytest.

## Setup

```bash
git clone https://github.com/SAMARTHJAISWAL/AsanaAbhyas.git
cd AsanaAbhyas
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                      # requirements.txt for runtime only
```

The first run downloads the pose model (about 9 MB for `full`, 30 MB for `heavy`), so it needs
internet access once. To work offline, pass a local `.task` file with `--model-path`.

## Usage

Command line:

```bash
python -m asana_abhyas --image photo.jpg                       # prints the pose, opens a window
python -m asana_abhyas --image photo.jpg --output out.jpg --no-display
python -m asana_abhyas --video clip.mp4 --output out.mp4 --no-display
python -m asana_abhyas --camera 0                              # webcam; ESC or q to quit
python -m asana_abhyas --help
```

Notebook:

```bash
mkdir media        # put your own full-body photos here (not included in the repo)
jupyter notebook AsanaAbhyas.ipynb
```

The notebook reads every image in `media/` (or `$ASANA_MEDIA_DIR`) and shows landmarks, a 3D
plot of the world landmarks, the joint angles, and the classified pose. The webcam cell is off
by default (`RUN_WEBCAM = False`).

Tests:

```bash
python -m pytest -q
```

## Classification rules

Angles are in degrees, measured at the middle joint (see `asana_abhyas/angles.py`).

| Pose | Elbows | Shoulders | Knees |
|---|---|---|---|
| Warrior II | both 165-195 | both 75-115 | one 165-195, other 90-120 |
| T Pose | both 165-195 | both 75-115 | both 160-195 |
| Tree Pose | any | any | one 165-195, other 315-335 (left) or 25-45 (right) |

Anything else is reported as `Unknown Pose`.

## Project structure

```
AsanaAbhyas.ipynb        walkthrough notebook (outputs cleared; bring your own images)
asana_abhyas/
  detector.py            PoseLandmarker wrapper + model download
  angles.py              calculate_angle()
  classifier.py          joint_angles(), classify_angles(), classify_pose()
  landmarks.py           33 landmark indices and skeleton connections
  drawing.py             OpenCV drawing of the skeleton and label
  __main__.py            CLI for image / video / webcam
tests/                   pytest suite for angles and rules
```

## Limitations

- The rules are hand-tuned thresholds, not a trained classifier. They only work well when the
  person faces the camera directly. The shoulder band is 75-115 degrees (wider than the
  tutorial's 80-110) because a Warrior II photo with a slightly leaning torso measured 112.
- Only one person per frame is analysed.
- No sample images are shipped. Use your own photos.

## Credits

The notebook's approach and structure (angle heuristics for Warrior II, T Pose and Tree Pose on
top of MediaPipe Pose) follow Bleed AI's public tutorial on real-time pose detection and
classification with MediaPipe. The original notebook used the now-removed `mp.solutions.pose` API.
It has been ported to the MediaPipe Tasks API and refactored into a package.
