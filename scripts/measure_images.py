#!/usr/bin/env python3
"""Compare saved RGB images using SSIM, PSNR and AlexNet LPIPS."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch

from deep_protect.perceptual import make_lpips


def rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("protected", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    a, b = rgb(args.reference), rgb(args.protected)
    if a.shape != b.shape:
        parser.error(f"Image dimensions differ: {a.shape} versus {b.shape}.")

    def tensor(image):
        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=args.device, dtype=torch.float32
        ).div(127.5).sub(1)

    with torch.no_grad():
        model = make_lpips("alex", args.device)
        lpips = model(tensor(a), tensor(b)).item()
    psnr = math.inf if np.array_equal(a, b) else peak_signal_noise_ratio(a, b, data_range=255)
    result = {
        "reference": str(args.reference), "protected": str(args.protected),
        "ssim": float(structural_similarity(a, b, data_range=255, channel_axis=-1,
                                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
        "psnr_db": float(psnr) if math.isfinite(psnr) else "Infinity",
        "lpips_alex": lpips,
    }
    report = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
