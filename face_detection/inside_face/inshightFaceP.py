from __future__ import annotations

import argparse

from application import FaceDetectionApplication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InsightFace image face detection")
    parser.add_argument("--images", help="Path to an image directory")
    parser.add_argument("--window-name", default="Face Detection Result", help="OpenCV window name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = FaceDetectionApplication(image_dir=args.images, window_name=args.window_name)
    app.run_images()


if __name__ == "__main__":
    main()