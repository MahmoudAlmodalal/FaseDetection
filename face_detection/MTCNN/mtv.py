from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "video" / "v8.mp4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy MTCNN + Norfair video tracking demo.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to the input video.")
    parser.add_argument("--window-name", default="Tracked Objects", help="OpenCV window title.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        global cv2
        import cv2
        import numpy as np
        from mtcnn import MTCNN
        from norfair import Detection, Tracker, draw_points
    except ImportError as exc:
        raise SystemExit("mtcnn and norfair are required for this experiment. Install them from requirements-experiments.txt.") from exc

    video_path = Path(args.video).expanduser()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    detector = MTCNN()
    tracker = Tracker(distance_function="euclidean", distance_threshold=20)

    try:
        while True:
            timer = cv2.getTickCount()
            ok, frame = capture.read()
            if not ok:
                break

            detections = []
            for face in detector.detect_faces(frame):
                x, y, w, h = face["box"]
                detections.append(Detection(np.array([x + w / 2.0, y + h / 2.0])))

            tracked_objects = tracker.update(detections=detections)
            frame_with_tracks = draw_points(frame.copy(), tracked_objects)
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
