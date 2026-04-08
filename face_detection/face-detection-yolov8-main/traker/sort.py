from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = PROJECT_ROOT / "video" / "v7.mp4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy SORT ROI tracking demo.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to the input video.")
    parser.add_argument("--window-name", default="Object Tracking", help="OpenCV window title.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    global cv2
    import cv2
    import numpy as np
    from sortt import Sort

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

    tracker = Sort()
    detection = np.array([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], 1.0]])

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            tracks = tracker.update(detection)
            for track in tracks:
                x1, y1, x2, y2 = [int(value) for value in track[:4]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow(args.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
