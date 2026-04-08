# Face Detection Workspace

This repository is a cleaned face-detection workspace with one supported pipeline and several legacy experiments.

## Supported Pipeline

The maintained code lives in `face_detection/inside_face`.

- Package name: `inside_face`
- Focus: InsightFace detection + Norfair tracking
- Docs: see [`face_detection/README.md`](face_detection/README.md)

From the repo root:

```bash
cd face_detection
python3 -m inside_face video --video /path/to/video.mp4
python3 -m inside_face video --profile tight --video /path/to/video.mp4
python3 -m inside_face images --images test_image
```

## Supported Vs Legacy

- Supported: `face_detection/inside_face`
- Legacy experiments: `face_detection/openCV`, `face_detection/MTCNN`, `face_detection/face-detection-yolov8-main`, `face_detection/dlib`, `face_detection/mediapipe`, `face_detection/p`, `yolo1`, and `d.py`

Legacy folders are kept for reference and experimentation. They now use repo-relative defaults and CLI arguments, but they are not maintained as stable interfaces.

## Installation

Install the supported pipeline:

```bash
python3 -m pip install -r requirements.txt
```

Install optional dependencies for the legacy experiments:

```bash
python3 -m pip install -r requirements-experiments.txt
```

`d.py` also needs a platform-specific Detectron2 install from the official Detectron2 instructions.

## Asset Policy

The repository intentionally ignores local videos, model weights, ONNX/PB files, executable artifacts, and the checked-in local Python environment under `face_detection/Lib/`.

- Keep sample images in `face_detection/test_image/` for lightweight demos.
- Keep large videos and model files local on your machine.
- Pass custom paths with CLI flags when running experiments.

## Experiment Catalog

| Path | Status | Purpose | Example run |
| --- | --- | --- | --- |
| `face_detection/inside_face` | Supported | InsightFace + Norfair package CLI | `cd face_detection && python3 -m inside_face video --video video/v1.mp4` |
| `face_detection/openCV/openCV.py` | Legacy | Haar cascade image detection | `python3 face_detection/openCV/openCV.py --images face_detection/test_image` |
| `face_detection/openCV/MTCNN_openCV.py` | Legacy | Compare MTCNN vs Haar cascade | `python3 face_detection/openCV/MTCNN_openCV.py --images face_detection/test_image` |
| `face_detection/MTCNN/MTCNN.py` | Legacy | MTCNN image detection | `python3 face_detection/MTCNN/MTCNN.py --images face_detection/test_image` |
| `face_detection/MTCNN/mv.py` | Legacy | MTCNN + MOSSE video tracking | `python3 face_detection/MTCNN/mv.py --video face_detection/video/v4.mp4` |
| `face_detection/MTCNN/mtv.py` | Legacy | MTCNN + Norfair tracking | `python3 face_detection/MTCNN/mtv.py --video face_detection/video/v8.mp4` |
| `face_detection/face-detection-yolov8-main/yolo.py` | Legacy | YOLOv8 + Norfair tracking | `python3 face_detection/face-detection-yolov8-main/yolo.py --video face_detection/video/v4.mp4` |
| `face_detection/face-detection-yolov8-main/m.py` | Legacy | YOLOv8 detection demo | `python3 face_detection/face-detection-yolov8-main/m.py --video face_detection/video/v8.mp4` |
| `face_detection/face-detection-yolov8-main/test_web.py` | Legacy | Heuristic face ID tracking | `python3 face_detection/face-detection-yolov8-main/test_web.py --video face_detection/video/v1.mp4` |
| `face_detection/face-detection-yolov8-main/traker/*.py` | Legacy | Tracker experiments with manual ROI selection | `python3 face_detection/face-detection-yolov8-main/traker/cvCSRT.py --video face_detection/video/v5.mp4` |
| `face_detection/p/faceTraking.py` | Legacy | Haar cascade + MOSSE tracking | `python3 face_detection/p/faceTraking.py --video face_detection/video/v8.mp4` |
| `face_detection/p/f.py` | Legacy | OpenCV MultiTracker experiment | `python3 face_detection/p/f.py --video face_detection/video/v4.mp4` |
| `face_detection/p/o.py` | Legacy | Manual ROI MOSSE demo | `python3 face_detection/p/o.py --video face_detection/video/v7.mp4` |
| `face_detection/dlib/t2.py` | Legacy | dlib image detection | `python3 face_detection/dlib/t2.py --image face_detection/test_image/m1.jpeg` |
| `face_detection/mediapipe/mediapipe1.py` | Legacy | face_recognition comparison | `python3 face_detection/mediapipe/mediapipe1.py --reference face_detection/test_image/m1.jpeg` |
| `yolo1/yolo.py` | Legacy | YOLO face video demo | `python3 yolo1/yolo.py --video face_detection/video/v4.mp4` |
| `d.py` | Legacy | Detectron2 image detection | `python3 d.py --image face_detection/test_image/m1.jpeg` |

## Before Pushing

Use this checklist before creating your first commit:

```bash
git init
git status --short
```

Expected result:

- Code, README files, and sample images appear as trackable files.
- Videos, weights, ONNX/PB files, executables, and `face_detection/Lib/` stay out of `git status`.

If you want the supported CLI to run from a clean shell, start from the `face_detection/` directory so `python3 -m inside_face ...` resolves correctly.
