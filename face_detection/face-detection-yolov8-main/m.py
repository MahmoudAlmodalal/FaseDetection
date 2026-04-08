from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "video" / "v8.mp4"
DEFAULT_WEIGHTS = Path(__file__).resolve().with_name("yolov8n-face.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy YOLOv8 face detection video demo.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to the input video.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to the YOLO weights file.")
    parser.add_argument("--window-name", default="YOLOv8 Face Detection", help="OpenCV window title.")
    return parser


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

    try:
        while True:
            timer = cv2.getTickCount()
            ok, frame = capture.read()
            if not ok:
                break

            for result in model(frame):
                for box in result.boxes.cpu().numpy():
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            fps = cv2.getTickFrequency() / max(cv2.getTickCount() - timer, 1)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(args.window_name, cv2.resize(frame, (1000, 650), interpolation=cv2.INTER_AREA))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
