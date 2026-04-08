from __future__ import annotations

import argparse

VIDEO_PROFILES = {
    "default": 60.0,
    "tight": 20.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the supported InsightFace + Norfair face detection pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    video_parser = subparsers.add_parser("video", help="Run face tracking on a video file.")
    video_parser.add_argument("--video", help="Path to the input video.")
    video_parser.add_argument(
        "--profile",
        choices=sorted(VIDEO_PROFILES),
        default="default",
        help="Tracking profile preset. Use 'tight' for stricter face association.",
    )
    video_parser.add_argument("--scale", type=float, default=0.25, help="Resize scale for video frames.")
    video_parser.add_argument(
        "--distance",
        type=float,
        help="Override the tracker distance threshold. Defaults to the selected profile value.",
    )
    video_parser.add_argument("--window-name", default="Face Detection", help="OpenCV window name.")

    images_parser = subparsers.add_parser("images", help="Run face detection on a directory of images.")
    images_parser.add_argument("--images", help="Path to an image directory.")
    images_parser.add_argument("--window-name", default="Face Detection Result", help="OpenCV window name.")

    return parser


def run_video(args: argparse.Namespace) -> None:
    from .application import FaceDetectionApplication

    distance = args.distance if args.distance is not None else VIDEO_PROFILES[args.profile]
    app = FaceDetectionApplication(
        video_path=args.video,
        resize_scale=args.scale,
        tracker_distance_threshold=distance,
        window_name=args.window_name,
    )
    app.run_video()


def run_images(args: argparse.Namespace) -> None:
    from .application import FaceDetectionApplication

    app = FaceDetectionApplication(image_dir=args.images, window_name=args.window_name)
    app.run_images()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "video":
        run_video(args)
        return

    if args.command == "images":
        run_images(args)
        return

    parser.error(f"Unsupported command: {args.command}")
