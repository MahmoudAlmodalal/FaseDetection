from __future__ import annotations

import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "test_image"


def get_image_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def show_face_detection(image, faces, elapsed_time: float, count: int) -> None:
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    for face in faces:
        x, y, width, height = face["box"]
        ax.add_patch(Rectangle((x, y), width, height, fill=False, color="red"))
    plt.text(10, -10, f"Elapsed time: {elapsed_time:.2f} seconds, {count} faces", fontsize=12, color="blue")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy MTCNN image detection demo.")
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

    detector = MTCNN(min_face_size=20, steps_threshold=[0.513, 0.513, 0.513], scale_factor=0.7)
    for image_path in get_image_paths(image_dir):
        start_time = time.time()
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping invalid image: {image_path}")
            continue
        faces = detector.detect_faces(image)
        show_face_detection(image, faces, time.time() - start_time, len(faces))


if __name__ == "__main__":
    main()
