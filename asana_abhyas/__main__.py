"""CLI: classify yoga poses in an image, a video file or a webcam feed.

    python -m asana_abhyas --image photo.jpg [--output out.jpg] [--no-display]
    python -m asana_abhyas --video clip.mp4 [--output out.mp4] [--no-display]
    python -m asana_abhyas --camera 0
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

from .classifier import classify_pose
from .detector import MODELS, PoseDetector
from .drawing import draw_label, draw_landmarks


def annotate(frame, detector):
    points, _ = detector.detect(frame)
    if points is None:
        return draw_label(frame, "No person detected"), None, None
    label, angles = classify_pose(points)
    draw_landmarks(frame, points)
    draw_label(frame, label)
    return frame, label, angles


def run_image(args) -> int:
    image = cv2.imread(args.image)
    if image is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        return 1
    with PoseDetector(args.model, args.model_path, video=False, min_confidence=args.min_confidence) as det:
        out, label, angles = annotate(image, det)
    print(f"Pose: {label or 'No person detected'}")
    if angles:
        print("Angles: " + ", ".join(f"{k}={v:.1f}" for k, v in vars(angles).items()))
    if args.output:
        cv2.imwrite(args.output, out)
        print(f"Saved {args.output}")
    if not args.no_display:
        cv2.imshow("AsanaAbhyas", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_stream(args) -> int:
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open {'video' if args.video else 'camera'}: {source}", file=sys.stderr)
        return 1
    writer = None
    prev = 0.0
    with PoseDetector(args.model, args.model_path, video=True, min_confidence=args.min_confidence) as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.video is None:
                frame = cv2.flip(frame, 1)  # selfie view for webcams
            frame, _, _ = annotate(frame, det)
            now = time.time()
            if prev:
                cv2.putText(frame, f"FPS: {int(1 / max(now - prev, 1e-6))}", (10, 60),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            prev = now
            if args.output:
                if writer is None:
                    h, w = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("AsanaAbhyas", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):  # ESC or q quits
                    break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="asana_abhyas", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", help="classify a single image")
    src.add_argument("--video", help="classify a video file")
    src.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--model", choices=MODELS, default="full", help="pose model size (default full)")
    p.add_argument("--model-path", help="use a local .task model file instead of downloading")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--output", help="save the annotated image/video here")
    p.add_argument("--no-display", action="store_true", help="do not open a window")
    args = p.parse_args(argv)
    return run_image(args) if args.image else run_stream(args)


if __name__ == "__main__":
    raise SystemExit(main())
