import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg

if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")
