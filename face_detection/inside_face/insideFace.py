from __future__ import annotations

import argparse

from application import FaceDetectionApplication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InsightFace + Norfair video face tracking")
    parser.add_argument("--video", help="Path to the input video")
    parser.add_argument("--scale", type=float, default=0.25, help="Resize scale for video frames")
    parser.add_argument("--distance", type=float, default=60.0, help="Tracker distance threshold")
    parser.add_argument("--window-name", default="Face Detection", help="OpenCV window name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = FaceDetectionApplication(
        video_path=args.video,
        resize_scale=args.scale,
        tracker_distance_threshold=args.distance,
        window_name=args.window_name,
    )
    app.run_video()


if __name__ == "__main__":
    main()