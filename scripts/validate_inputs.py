#!/usr/bin/env python3
"""Validate trusted checkpoints using their production loaders."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from deep_protect.diagnostics import RunProfile
from deep_protect.feature_bank import load_feature_bank
from deep_protect.models.arcface import extract_arcface, load_arcface
from deep_protect.models.e4e import E4EEncoder
from deep_protect.models.farl import FaRLTextImageEncoder
from deep_protect.models.stylegan import load_stylegan
from deep_protect.perceptual import make_lpips


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_tensor(value, shape):
    if tuple(value.shape) != shape or not torch.isfinite(value).all():
        raise ValueError(f"Expected finite {shape}, got {tuple(value.shape)}")


@torch.no_grad()
def probe(name, path, device):
    if name == "stylegan":
        model = load_stylegan(path, device=device)
        if (model.num_ws, model.w_dim, model.img_resolution, model.img_channels) != (18, 512, 1024, 3):
            raise ValueError("Expected the FFHQ 1024 StyleGAN2-ADA generator.")
        latent = torch.zeros(1, 18, 512, device=device)
        check_tensor(model.synthesis(latent, noise_mode="const", force_fp32=True), (1, 3, 1024, 1024))
    elif name == "farl":
        model = FaRLTextImageEncoder(path, device=device)
        check_tensor(model.encode_text("nose"), (1, 512))
        check_tensor(model.encode_images([Image.new("RGB", (224, 224))]), (1, 512))
    elif name == "e4e":
        model = E4EEncoder(path, device=device)
        check_tensor(model.encode(torch.zeros(1, 3, 256, 256, device=device)), (1, 18, 512))
    elif name == "arcface":
        model = load_arcface(path, device=device)
        check_tensor(extract_arcface(model, torch.zeros(1, 3, 256, 256, device=device)), (1, 512))
    else:
        model = make_lpips("alex", device)
        check_tensor(model(torch.zeros(1, 3, 64, 64, device=device),
                           torch.zeros(1, 3, 64, 64, device=device)), (1, 1, 1, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stylegan", "e4e", "arcface", "farl", "feature-bank"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--expected-identities", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trusted-inputs", action="store_true",
                        help="Acknowledge that checkpoint and legacy NPZ deserialization can execute code")
    args = parser.parse_args()
    names = ("stylegan", "e4e", "arcface", "farl")
    checks = [(name, getattr(args, name)) for name in names if getattr(args, name)]
    bank_path = args.feature_bank
    if not (checks or bank_path or args.lpips):
        parser.error("Supply a checkpoint, --feature-bank, or --lpips.")
    if (checks or (bank_path and bank_path.suffix.lower() == ".npz")) and not args.trusted_inputs:
        parser.error("Use --trusted-inputs for files obtained from a trusted source.")
    if args.candidate_count < 1:
        parser.error("--candidate-count must be positive.")
    if args.lpips:
        checks.append(("lpips", None))
    profile = RunProfile(args.device)
    for name, path in checks:
        if path:
            print(json.dumps({"model": name, "path": str(path), "sha256": checksum(path)}), flush=True)
        with profile.stage(name):
            probe(name, path, args.device)
    if bank_path:
        bank = load_feature_bank(bank_path)
        if bank.size < 2 * args.candidate_count:
            raise ValueError(f"Attribute mode needs at least {2 * args.candidate_count} rows.")
        if args.expected_identities is not None and bank.size != args.expected_identities:
            raise ValueError(f"Expected {args.expected_identities} bank rows, found {bank.size}.")
        if bank.names and len(set(bank.names)) != bank.size:
            raise ValueError("Duplicate gallery paths found.")
        print(json.dumps({"bank_rows": bank.size, "sha256": checksum(bank_path)}))


if __name__ == "__main__":
    main()
