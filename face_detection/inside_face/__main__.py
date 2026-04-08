try:
    from .insideFace import main
except ImportError:  # pragma: no cover - script execution fallback
    from insideFace import main


if __name__ == "__main__":
    main()