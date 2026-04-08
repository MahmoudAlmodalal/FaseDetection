# inside_face Pipeline

This directory contains the supported face-detection pipeline for the workspace. The package is `inside_face`, and it is the most structured part of the repository.

## Layout

- `inside_face/domain.py`: shared data models and runtime settings
- `inside_face/config.py`: project-relative defaults and path resolution
- `inside_face/infrastructure.py`: detector, tracker, video source, and rendering utilities
- `inside_face/application.py`: orchestration layer for image and video flows
- `inside_face/cli.py`: subcommand CLI used by `python -m inside_face`

## Dependencies

Install the supported dependencies from the repository root:

```bash
python3 -m pip install -r ../requirements.txt
```

## Run

From this directory:

```bash
python3 -m inside_face video --video video/v1.mp4
python3 -m inside_face video --profile tight --video video/v1.mp4
python3 -m inside_face images --images test_image
```

Or from the repository root:

```bash
cd face_detection
python3 -m inside_face video --video video/v1.mp4
```

## Notes

- The CLI uses subcommands instead of the older standalone entry scripts.
- Default media resolution is taken from `video/` and `test_image/` when you do not pass custom paths.
- `tight` is a preset for stricter tracker matching in video mode.
- Large videos and model files are intentionally ignored by git in the workspace root.
