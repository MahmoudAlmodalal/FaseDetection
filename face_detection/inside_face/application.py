from __future__ import annotations

import time

import cv2

from .config import build_settings
from .infrastructure import (
    FrameRenderer,
    InsightFaceDetector,
    NorfairFaceTracker,
    OpenCVVideoSource,
    SUPPORTED_IMAGE_SUFFIXES,
)


class FaceDetectionApplication:
    def __init__(
        self,
        video_path: str | None = None,
        image_dir: str | None = None,
        resize_scale: float = 0.25,
        tracker_distance_threshold: float = 60.0,
        tracker_detection_threshold: float = 1.0,
        tracker_reid_distance_threshold: float = 5.0,
        tracker_reid_hit_counter_max: int = 0,
        window_name: str = "Face Detection",
    ) -> None:
        self.settings = build_settings(
            video_path=video_path,
            image_dir=image_dir,
            resize_scale=resize_scale,
            window_name=window_name,
            tracker_distance_threshold=tracker_distance_threshold,
            tracker_detection_threshold=tracker_detection_threshold,
            tracker_reid_distance_threshold=tracker_reid_distance_threshold,
            tracker_reid_hit_counter_max=tracker_reid_hit_counter_max,
        )
        self.detector = InsightFaceDetector(self.settings)
        self.tracker = NorfairFaceTracker(self.settings)
        self.renderer = FrameRenderer(self.settings.window_name)

    def _resize_frame(self, frame: cv2.Mat) -> cv2.Mat:
        if self.settings.resize_scale == 1:
            return frame
        return cv2.resize(
            frame,
            (int(frame.shape[1] * self.settings.resize_scale), int(frame.shape[0] * self.settings.resize_scale)),
            interpolation=cv2.INTER_AREA,
        )

    def run_video(self) -> None:
        source = OpenCVVideoSource(self.settings.video_path)
        try:
            while True:
                start = time.perf_counter()
                ok, frame = source.read()
                if not ok:
                    break

                frame = self._resize_frame(frame)
                detections = self.detector.detect(frame)
                tracked_objects = self.tracker.update(detections)
                fps = 1.0 / max(time.perf_counter() - start, 1e-6)
                rendered = self.renderer.render_video_frame(frame, detections, tracked_objects, fps)

                cv2.imshow(self.settings.window_name, rendered)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            source.release()
            cv2.destroyAllWindows()

    def run_images(self) -> None:
        image_paths = sorted(
            path
            for path in self.settings.image_dir.iterdir()
            if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )

        if not image_paths:
            raise FileNotFoundError(f"No supported images found in: {self.settings.image_dir}")

        for image_path in image_paths:
            start = time.perf_counter()
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image = cv2.resize(image, (1000, 650), interpolation=cv2.INTER_AREA)
            detections = self.detector.detect(image)
            elapsed = time.perf_counter() - start
            rendered = self.renderer.render_image_frame(image, detections, elapsed)
            cv2.imshow(self.settings.window_name, rendered)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
