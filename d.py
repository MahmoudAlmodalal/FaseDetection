import torch
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from PIL import Image
import numpy as np

# Load a pre-trained Detectron2 model for face detection
cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for detection confidence
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml")
predictor = DefaultPredictor(cfg)

# Load and process an input image
image = np.array(Image.open("C:/Users/p8036/Desktop/face detection/face_detection/test_image/m1.jpeg"))
outputs = predictor(image)

# Access detected faces' bounding boxes and classes
instances = outputs["instances"]
faces = instances[instances.pred_classes == 0]  # Assuming face class is 0 in COCO dataset
bounding_boxes = faces.pred_boxes.tensor.cpu().numpy()

# Print the bounding boxes
for bbox in bounding_boxes:
    x1, y1, x2, y2 = bbox
    print(f"Face detected at coordinates: ({x1}, {y1}, {x2}, {y2})")
