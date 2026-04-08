from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "video" / "v7.mp4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy manual ROI MOSSE tracking demo.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to the input video.")
    parser.add_argument("--window-name", default="frame", help="OpenCV window title.")
    return parser


def tracker_factory():
    if hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerMOSSE_create"):
        return legacy.TrackerMOSSE_create
    raise RuntimeError("MOSSE tracker is not available in this OpenCV build.")


def main() -> None:
    global cv2
    import cv2

    args = build_parser().parse_args()
    video_path = Path(args.video).expanduser()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise SystemExit(f"Could not read the first frame from: {video_path}")

    frame = cv2.resize(frame, (1000, 650), interpolation=cv2.INTER_AREA)
    bbox = cv2.selectROI(args.window_name, frame, False)
    if bbox == (0, 0, 0, 0):
        capture.release()
        cv2.destroyAllWindows()
        raise SystemExit("ROI selection was cancelled.")

    tracker = tracker_factory()()
    tracker.init(frame, bbox)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = cv2.resize(frame, (1000, 650), interpolation=cv2.INTER_AREA)
            tracked, track_window = tracker.update(frame)
            if tracked:
                x, y, w, h = [int(value) for value in track_window]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow(args.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
