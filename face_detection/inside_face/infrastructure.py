from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError(
        "insightface is required for the face detection pipeline. Install it before running this project."
    ) from exc

try:
    from norfair import Detection, Tracker as NorfairTracker, draw_points
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError(
        "norfair is required for the tracking pipeline. Install it before running this project."
    ) from exc

from .domain import FaceBox, FaceTrackingSettings

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class InsightFaceDetector:
    def __init__(self, settings: FaceTrackingSettings) -> None:
        self.settings = settings
        self.model = FaceAnalysis(allowed_modules=["detection"])
        self.model.prepare(ctx_id=settings.ctx_id, det_size=settings.det_size)

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        detected_faces = self.model.get(frame)
        boxes: list[FaceBox] = []
        for face in detected_faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            boxes.append(FaceBox(x1=x1, y1=y1, x2=x2, y2=y2))
        return boxes


class NorfairFaceTracker:
    def __init__(self, settings: FaceTrackingSettings) -> None:
        self.settings = settings
        self.tracker = NorfairTracker(
            distance_function="euclidean",
            distance_threshold=settings.tracker_distance_threshold,
            detection_threshold=settings.tracker_detection_threshold,
            reid_distance_threshold=settings.tracker_reid_distance_threshold,
            reid_hit_counter_max=settings.tracker_reid_hit_counter_max,
        )

    def update(self, detections: Iterable[FaceBox]):
        norfair_detections = [Detection(np.array(face.center)) for face in detections]
        return self.tracker.update(detections=norfair_detections)


class OpenCVVideoSource:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

    def read(self):
        return self.capture.read()

    def frame_size(self) -> tuple[int, int]:
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def release(self) -> None:
        self.capture.release()


class FrameRenderer:
    def __init__(self, window_name: str) -> None:
        self.window_name = window_name

    def _draw_boxes(self, frame: np.ndarray, detections: Iterable[FaceBox]) -> np.ndarray:
        annotated = frame.copy()
        for box in detections:
            cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
        return annotated

    def _draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"FPS: {int(fps)}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        return annotated

    def render_video_frame(
        self,
        frame: np.ndarray,
        detections: Iterable[FaceBox],
        tracked_objects,
        fps: float,
    ) -> np.ndarray:
        annotated = self._draw_boxes(frame, detections)
        annotated = draw_points(annotated, tracked_objects, text_size=0.75)
        return self._draw_fps(annotated, fps)

    def render_image_frame(self, frame: np.ndarray, detections: Iterable[FaceBox], elapsed_seconds: float) -> np.ndarray:
        detections = list(detections)
        annotated = self._draw_boxes(frame, detections)
        cv2.putText(
            annotated,
            f"time: {elapsed_seconds:.3f}s, faces: {len(detections)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return annotated
