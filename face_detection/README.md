# Face Detection Workspace

This workspace contains multiple face detection experiments. The most organized one is the cleaned InsightFace + Norfair pipeline in [insideFace/](insideFace).

## Clean architecture layout

- [insideFace/domain.py](insideFace/domain.py) - data models and settings
- [insideFace/config.py](insideFace/config.py) - path resolution and defaults
- [insideFace/infrastructure.py](insideFace/infrastructure.py) - detector, tracker, video source, and rendering
- [insideFace/application.py](insideFace/application.py) - orchestration layer
- [insideFace/insideFace.py](insideFace/insideFace.py) - video entrypoint
- [insideFace/inshightFaceP.py](insideFace/inshightFaceP.py) - image entrypoint
- [insideFace/inshightFaceMosse.py](insideFace/inshightFaceMosse.py) - tighter tracking variant

## Requirements

Install the Python packages used by the clean pipeline:

- opencv-python
- insightface
- norfair
- numpy

## Run

From the project root:

- Video tracking: `python insideFace/insideFace.py`
- Video tracking with package mode: `python -m insideFace`
- Image detection: `python insideFace/inshightFaceP.py --images /path/to/images`

## Runtime notes

- The app looks for a default video in [video/](video) and default images in [test_image/](test_image).
- Use `--video` to point to a custom video file.
- Use `--scale` to control resizing speed vs. accuracy.
- Use `--distance` to tune tracker behavior.

## Maintainability goals

- keep entry files thin
- keep settings in one place
- use shared domain objects
- avoid duplicating detector/tracker logic across scripts