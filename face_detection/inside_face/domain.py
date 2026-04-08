from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class FaceTrackingSettings:
    video_path: Path
    image_dir: Path
    ctx_id: int = 0
    det_size: tuple[int, int] = (640, 640)
    resize_scale: float = 0.25
    window_name: str = "Face Detection"
    tracker_distance_threshold: float = 60.0
    tracker_detection_threshold: float = 1.0
    tracker_reid_distance_threshold: float = 5.0
    tracker_reid_hit_counter_max: int = 0
