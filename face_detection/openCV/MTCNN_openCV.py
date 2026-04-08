from __future__ import annotations

import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "test_image"


def get_image_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def show_face_detection(image, mtcnn_faces, cv_faces, elapsed_time: float) -> None:
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    for face in mtcnn_faces:
        x, y, width, height = face["box"]
        ax.add_patch(Rectangle((x, y), width, height, fill=False, color="red"))
    for x, y, width, height in cv_faces:
        ax.add_patch(Rectangle((x, y), width, height, fill=False, color="green"))
    plt.text(10, 0, f"Elapsed time: {elapsed_time:.2f} seconds", fontsize=12, color="blue")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy MTCNN vs OpenCV image comparison.")
    parser.add_argument("--images", default=str(DEFAULT_IMAGE_DIR), help="Directory containing demo images.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        global cv2
        import cv2
        from mtcnn import MTCNN
    except ImportError as exc:
        raise SystemExit("mtcnn is required for this experiment. Install it from requirements-experiments.txt.") from exc

    image_dir = Path(args.images).expanduser()
    if not image_dir.exists():
        raise SystemExit(f"Image directory not found: {image_dir}")

    detector = MTCNN(min_face_size=20, steps_threshold=[0.6, 0.7, 0.7], scale_factor=0.7)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for image_path in get_image_paths(image_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping invalid image: {image_path}")
            continue
        start_time = time.time()
        mtcnn_faces = detector.detect_faces(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
        show_face_detection(image, mtcnn_faces, cv_faces, time.time() - start_time)


if __name__ == "__main__":
    main()
