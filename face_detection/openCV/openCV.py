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
    for x, y, width, height in faces:
        ax.add_patch(Rectangle((x, y), width, height, fill=False, color="red"))
    plt.text(10, -10, f"Elapsed time: {elapsed_time:.2f} seconds, {count} faces", fontsize=12, color="blue")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy OpenCV Haar cascade image demo.")
    parser.add_argument("--images", default=str(DEFAULT_IMAGE_DIR), help="Directory containing demo images.")
    return parser


def main() -> None:
    global cv2
    import cv2

    args = build_parser().parse_args()
    image_dir = Path(args.images).expanduser()
    if not image_dir.exists():
        raise SystemExit(f"Image directory not found: {image_dir}")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise SystemExit("Could not load OpenCV Haar cascade.")

    for image_path in get_image_paths(image_dir):
        start_time = time.time()
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping invalid image: {image_path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(20, 20))
        show_face_detection(image, faces, time.time() - start_time, len(faces))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
