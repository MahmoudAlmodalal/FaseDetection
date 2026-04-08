from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "video" / "v4.mp4"
DEFAULT_WEIGHTS = Path(__file__).resolve().with_name("yolov8n-face.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy YOLOv8 + Norfair face tracking demo.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to the input video.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to the YOLO weights file.")
    parser.add_argument("--scale", type=float, default=0.25, help="Resize scale for video frames.")
    parser.add_argument("--distance", type=float, default=60.0, help="Norfair distance threshold.")
    parser.add_argument("--window-name", default="Tracked Objects", help="OpenCV window title.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        global cv2
        import cv2
        import numpy as np
        from norfair import Detection, Tracker, draw_points
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics and norfair are required for this experiment. Install them from requirements-experiments.txt."
        ) from exc

    video_path = Path(args.video).expanduser()
    weights_path = Path(args.weights).expanduser()

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=args.distance,
        detection_threshold=1,
        reid_distance_threshold=5,
        reid_hit_counter_max=0,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    try:
        while True:
            timer = cv2.getTickCount()
            ok, frame = capture.read()
            if not ok:
                break

            if args.scale != 1:
                frame = cv2.resize(
                    frame,
                    (int(frame.shape[1] * args.scale), int(frame.shape[0] * args.scale)),
                    interpolation=cv2.INTER_AREA,
                )

            detections = []
            for face in model(frame):
                for box in face.boxes.cpu().numpy():
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    detections.append(Detection(np.array([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2])))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            tracked_objects = tracker.update(detections=detections)
            frame_with_tracks = draw_points(frame, tracked_objects, text_size=0.75)
            fps = cv2.getTickFrequency() / max(cv2.getTickCount() - timer, 1)
            cv2.putText(frame_with_tracks, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(args.window_name, frame_with_tracks)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
