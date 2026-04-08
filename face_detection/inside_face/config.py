from __future__ import annotations

from pathlib import Path

from .domain import FaceTrackingSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "test_image"
DEFAULT_VIDEO_CANDIDATES = (
    PROJECT_ROOT / "video" / "v11.mp4",
    PROJECT_ROOT / "video" / "v10.mp4",
)


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No valid media file was found. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def build_settings(
    video_path: str | None = None,
    image_dir: str | None = None,
    resize_scale: float = 0.25,
    window_name: str = "Face Detection",
    tracker_distance_threshold: float = 60.0,
    tracker_detection_threshold: float = 1.0,
    tracker_reid_distance_threshold: float = 5.0,
    tracker_reid_hit_counter_max: int = 0,
) -> FaceTrackingSettings:
    resolved_video = Path(video_path).expanduser() if video_path else resolve_existing_path(*DEFAULT_VIDEO_CANDIDATES)
    resolved_image_dir = Path(image_dir).expanduser() if image_dir else DEFAULT_IMAGE_DIR

    if not resolved_video.exists():
        raise FileNotFoundError(f"Video file not found: {resolved_video}")
    if not resolved_image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {resolved_image_dir}")

    return FaceTrackingSettings(
        video_path=resolved_video,
        image_dir=resolved_image_dir,
        resize_scale=resize_scale,
        window_name=window_name,
        tracker_distance_threshold=tracker_distance_threshold,
        tracker_detection_threshold=tracker_detection_threshold,
        tracker_reid_distance_threshold=tracker_reid_distance_threshold,
        tracker_reid_hit_counter_max=tracker_reid_hit_counter_max,
    )
