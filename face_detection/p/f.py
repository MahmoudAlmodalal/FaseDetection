from __future__ import annotations

import argparse
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "video" / "v4.mp4"
DEFAULT_CASCADE = Path(__file__).resolve().with_name("h1.xml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy multi-tracker Haar cascade demo.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to the input video.")
    parser.add_argument("--cascade", default=str(DEFAULT_CASCADE), help="Path to the Haar cascade XML file.")
    parser.add_argument("--window-name", default="Multi-Object Tracking", help="OpenCV window title.")
    return parser


def tracker_factory():
    if hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerMOSSE_create"):
        return legacy.TrackerMOSSE_create
    raise RuntimeError("MOSSE tracker is not available in this OpenCV build.")


def multi_tracker_factory():
    if hasattr(cv2, "MultiTracker_create"):
        return cv2.MultiTracker_create
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "MultiTracker_create"):
        return legacy.MultiTracker_create
    raise RuntimeError("MultiTracker is not available in this OpenCV build.")


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def draw_box(frame, bbox, tracker_id: int) -> None:
    x, y, w, h = [int(value) for value in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3, 1)
    cv2.putText(frame, f"Tracker ID: {tracker_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def main() -> None:
    global cv2
    import cv2

    args = build_parser().parse_args()
    video_path = Path(args.video).expanduser()
    cascade_path = Path(args.cascade).expanduser()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        capture.release()
        raise SystemExit(f"Could not load cascade file: {cascade_path}")

    create_tracker = tracker_factory()
    multi_tracker = multi_tracker_factory()()

    try:
        while True:
            timer = cv2.getTickCount()
            ok, frame = capture.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.06, minNeighbors=5, minSize=(20, 20))
            tracked_boxes = list(multi_tracker.getObjects())

            for face in faces:
                bbox = tuple(face)
                already_tracked = any(calculate_distance(bbox[0], bbox[1], tracked[0], tracked[1]) < 50 for tracked in tracked_boxes)
                if not already_tracked:
                    multi_tracker.add(create_tracker(), frame, bbox)
                    tracked_boxes.append(bbox)

            _, boxes = multi_tracker.update(frame)
            for index, bbox in enumerate(boxes, start=1):
                draw_box(frame, bbox, index)

            fps = cv2.getTickFrequency() / max(cv2.getTickCount() - timer, 1)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(args.window_name, cv2.resize(frame, (1000, 650), interpolation=cv2.INTER_AREA))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
