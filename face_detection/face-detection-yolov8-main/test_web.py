from __future__ import annotations

import argparse
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "video" / "v1.mp4"
DEFAULT_WEIGHTS = Path(__file__).resolve().with_name("yolov8n-face.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy YOLOv8 heuristic ID tracking demo.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to the input video.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to the YOLO weights file.")
    parser.add_argument("--window-name", default="YOLOv8 Object Detection", help="OpenCV window title.")
    return parser


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def draw_box(frame, bbox, face_id: int) -> None:
    cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 2)
    cv2.putText(frame, f"Face ID: {face_id}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


def main() -> None:
    args = build_parser().parse_args()

    try:
        global cv2
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required for this experiment. Install it from requirements-experiments.txt.") from exc

    video_path = Path(args.video).expanduser()
    weights_path = Path(args.weights).expanduser()

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    face_id_counter = 1
    face_to_tracker_map: dict[int, tuple[int, int, int, int]] = {}

    try:
        while True:
            timer = cv2.getTickCount()
            ok, frame = capture.read()
            if not ok:
                break

            for result in model(frame):
                for box in result.boxes.cpu().numpy():
                    bbox = tuple(box.xyxy[0].astype(int))
                    matched_face_id = None

                    for face_id, known_bbox in face_to_tracker_map.items():
                        center_x = bbox[0] + (bbox[2] - bbox[0]) / 2
                        center_y = bbox[1] + (bbox[3] - bbox[1]) / 2
                        distance = calculate_distance(center_x, center_y, (known_bbox[0] + known_bbox[2]) / 2, (known_bbox[1] + known_bbox[3]) / 2)
                        if distance < frame.shape[1] * frame.shape[0] * 0.000017:
                            matched_face_id = face_id
                            break

                    if matched_face_id is None:
                        matched_face_id = face_id_counter
                        face_id_counter += 1

                    face_to_tracker_map[matched_face_id] = bbox
                    draw_box(frame, bbox, matched_face_id)

            fps = cv2.getTickFrequency() / max(cv2.getTickCount() - timer, 1)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(
                args.window_name,
                cv2.resize(frame, (int(frame.shape[1] * 0.3), int(frame.shape[0] * 0.3)), interpolation=cv2.INTER_AREA),
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
