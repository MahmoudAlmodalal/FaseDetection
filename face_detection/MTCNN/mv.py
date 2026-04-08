from __future__ import annotations

import argparse
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "video" / "v4.mp4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy MTCNN + MOSSE video tracking demo.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to the input video.")
    parser.add_argument("--detect-every", type=int, default=24, help="Run face detection every N frames.")
    parser.add_argument("--window-name", default="Multi-Object Tracking", help="OpenCV window title.")
    return parser


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def tracker_factory():
    if hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerMOSSE_create"):
        return legacy.TrackerMOSSE_create
    raise RuntimeError("MOSSE tracker is not available in this OpenCV build.")


def draw_box(frame, bbox, tracker_id: int, highlight: bool = False) -> None:
    color = (0, 255, 0) if highlight else (0, 0, 255)
    x, y, w, h = [int(value) for value in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3, 1)
    cv2.putText(frame, f"Tracker ID: {tracker_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def main() -> None:
    args = build_parser().parse_args()

    try:
        global cv2
        import cv2
        from mtcnn import MTCNN
    except ImportError as exc:
        raise SystemExit("mtcnn is required for this experiment. Install it from requirements-experiments.txt.") from exc

    video_path = Path(args.video).expanduser()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    detector = MTCNN(min_face_size=30, steps_threshold=[0.7, 0.7, 0.7], scale_factor=0.7)
    trackers: dict[int, tuple[object, tuple[int, int, int, int]]] = {}
    tracker_id_counter = 1
    face_to_tracker_map: dict[int, tuple[int, int, int, int]] = {}
    create_tracker = tracker_factory()
    frame_index = 0

    try:
        while True:
            timer = cv2.getTickCount()
            ok, frame = capture.read()
            if not ok:
                break

            stale_ids = []
            for tracker_id, (tracker, bbox) in list(trackers.items()):
                success, new_bbox = tracker.update(frame)
                if not success:
                    stale_ids.append(tracker_id)
                    continue
                trackers[tracker_id] = (tracker, new_bbox)
                draw_box(frame, new_bbox, tracker_id)

            for tracker_id in stale_ids:
                trackers.pop(tracker_id, None)

            if frame_index % max(args.detect_every, 1) == 0:
                for face in detector.detect_faces(frame):
                    x, y, w, h = face["box"]
                    bbox = (x, y, w, h)
                    face_id = None
                    for known_face_id, known_bbox in face_to_tracker_map.items():
                        if calculate_distance(bbox[0], bbox[1], known_bbox[0], known_bbox[1]) < 50:
                            face_id = known_face_id
                            break

                    if face_id is None:
                        face_id = tracker_id_counter
                        tracker_id_counter += 1
                        tracker = create_tracker()
                        tracker.init(frame, bbox)
                        trackers[face_id] = (tracker, bbox)

                    face_to_tracker_map[face_id] = bbox
                    draw_box(frame, bbox, face_id, highlight=True)

            fps = cv2.getTickFrequency() / max(cv2.getTickCount() - timer, 1)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(args.window_name, cv2.resize(frame, (1000, 650), interpolation=cv2.INTER_AREA))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
