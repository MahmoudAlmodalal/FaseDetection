from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = PROJECT_ROOT / "test_image" / "m1.jpeg"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "test_image"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy face comparison experiment using face_recognition.")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="Reference image path.")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGE_DIR), help="Directory containing comparison images.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        global cv2
        import cv2
        import face_recognition
    except ImportError as exc:
        raise SystemExit(
            "face_recognition is required for this experiment. Install it from requirements-experiments.txt."
        ) from exc

    reference_path = Path(args.reference).expanduser()
    image_dir = Path(args.images_dir).expanduser()

    if not reference_path.exists():
        raise SystemExit(f"Reference image not found: {reference_path}")
    if not image_dir.exists():
        raise SystemExit(f"Image directory not found: {image_dir}")

    reference_image = cv2.imread(str(reference_path))
    if reference_image is None:
        raise SystemExit(f"Could not read reference image: {reference_path}")

    reference_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    reference_encodings = face_recognition.face_encodings(reference_rgb)
    if not reference_encodings:
        raise SystemExit(f"No face found in reference image: {reference_path}")

    reference_encoding = reference_encodings[0]
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping invalid image: {image_path}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        if not encodings:
            print(f"No face found in {image_path.name}")
            continue

        result = face_recognition.compare_faces([reference_encoding], encodings[0])[0]
        print(f"{image_path.name}: {result}")
        cv2.imshow("reference", reference_image)
        cv2.imshow("candidate", image)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
