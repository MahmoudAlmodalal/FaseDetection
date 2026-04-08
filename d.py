from __future__ import annotations

import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = WORKSPACE_ROOT / "face_detection" / "test_image" / "m1.jpeg"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy Detectron2 image detection experiment.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Path to the input image.")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Minimum prediction confidence.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        import torch
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Detectron2 is required for this experiment. Install it from requirements-experiments.txt.") from exc

    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml")

    predictor = DefaultPredictor(cfg)
    image = np.array(Image.open(image_path))
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    people = instances[instances.pred_classes == 0]

    if len(people) == 0:
        print("No person detections were found in the image.")
        return

    for bbox in people.pred_boxes.tensor.numpy():
        x1, y1, x2, y2 = bbox
        print(f"Detection at coordinates: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")


if __name__ == "__main__":
    main()
