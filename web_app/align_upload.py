"""Decode an uploaded photo and run the existing FFHQ alignment."""

import sys
from pathlib import Path

import dlib
from PIL import Image, ImageOps

from scripts.align_faces import align_face


def main():
    upload, output, predictor = map(Path, sys.argv[1:])
    try:
        with Image.open(upload) as photo:
            if photo.format not in {"JPEG", "PNG"} or getattr(photo, "n_frames", 1) != 1:
                print("UPLOAD_ERROR:format", flush=True)
                return 2
            if photo.width * photo.height > 24_000_000:
                print("UPLOAD_ERROR:dimensions", flush=True)
                return 2
            ImageOps.exif_transpose(photo).convert("RGB").save(output / "original.png")
    except (OSError, ValueError, Image.DecompressionBombError):
        print("UPLOAD_ERROR:invalid", flush=True)
        return 2
    try:
        aligned = align_face(output / "original.png", dlib.shape_predictor(str(predictor)))
    except RuntimeError as exc:
        if str(exc) != "no face detected":
            raise
        print("UPLOAD_ERROR:face", flush=True)
        return 2
    aligned.save(output / "aligned.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
