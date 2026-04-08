from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "test_image" / "m3.jpg"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy DeepFace MTCNN image demo.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Path to the input image.")
    parser.add_argument("--detector-backend", default="mtcnn", help="DeepFace detector backend name.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        global cv2
        import cv2
        from deepface import DeepFace
    except ImportError as exc:
        raise SystemExit("deepface is required for this experiment. Install it from requirements-experiments.txt.") from exc

    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    faces = DeepFace.extract_faces(
        img_path=str(image_path),
        detector_backend=args.detector_backend,
        enforce_detection=False,
    )
    for face in faces:
        x = int(face["facial_area"]["x"])
        y = int(face["facial_area"]["y"])
        w = int(face["facial_area"]["w"])
        h = int(face["facial_area"]["h"])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
