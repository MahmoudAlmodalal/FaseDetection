from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = PROJECT_ROOT / "video" / "v5.mp4"


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to the input video.")
    parser.add_argument("--window-name", default="Object Tracking", help="OpenCV window title.")
    return parser


def _tracker_creator(name: str):
    import cv2

    creator_name = f"Tracker{name}_create"

    if hasattr(cv2, creator_name):
        return getattr(cv2, creator_name)

    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, creator_name):
        return getattr(legacy, creator_name)

    raise RuntimeError(f"OpenCV tracker '{name}' is not available in this environment.")


def run_tracker_demo(tracker_name: str, description: str) -> None:
    parser = build_parser(description)
    args = parser.parse_args()
    import cv2

    video_path = Path(args.video).expanduser()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise SystemExit(f"Could not read the first frame from: {video_path}")

    bbox = cv2.selectROI(args.window_name, frame, False)
    if bbox == (0, 0, 0, 0):
        capture.release()
        cv2.destroyAllWindows()
        raise SystemExit("ROI selection was cancelled.")

    tracker = _tracker_creator(tracker_name)()
    tracker.init(frame, bbox)

    try:
        while True:
            timer = cv2.getTickCount()
            ok, frame = capture.read()
            if not ok:
                break

            tracked, updated_bbox = tracker.update(frame)
            if tracked:
                x, y, w, h = [int(value) for value in updated_bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            fps = cv2.getTickFrequency() / max(cv2.getTickCount() - timer, 1)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(args.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
