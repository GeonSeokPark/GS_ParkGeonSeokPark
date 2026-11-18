"""Run the approved reconstruction and watermark sweep for a single photo."""

import argparse
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import perf_counter

from .settings import CONV_BLOCKS, LORA_BLOCKS, trials, validate

ROOT = Path(__file__).resolve().parents[1]


def event(stage, current=0, total=1):
    print("WEB " + json.dumps({"stage": stage, "current": current, "total": total}), flush=True)


def run_script(script, args, stage, total, pattern):
    event(stage, 0, total)
    command = [sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, args)]
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace", env=child_env,
    ) as child:
        for line in child.stdout:
            print(line, end="", flush=True)
            match = pattern.search(line)
            if match:
                event(stage, int(match[1]), total)
        if child.wait():
            raise RuntimeError(f"{script} failed with exit status {child.returncode}")
def sweep(args, refined, settings):
    import numpy as np
    import torch
    from PIL import Image
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    from deep_protect.feature_bank import load_feature_bank
    from deep_protect.models.arcface import extract_arcface, load_arcface
    from deep_protect.models.farl import FaRLTextImageEncoder
    from deep_protect.models.lora import inject_lora, load_lora_state_dict
    from deep_protect.models.stylegan import load_stylegan
    from deep_protect.perceptual import make_lpips
    from deep_protect.utils import pil_to_tensor, tensor_to_pil
    from deep_protect.watermark import adversarial_watermark, attribute_direction

    candidates_settings = trials(settings)
    total = len(candidates_settings)
    event("sweep", 0, total)
    checkpoint = args.models
    metadata = json.loads((refined / "metadata.json").read_text(encoding="utf-8"))
    device = "cuda"
    with torch.no_grad():
        generator = load_stylegan(checkpoint / "ffhq.pkl", device)
        inject_lora(generator, blocks=metadata["lora_blocks"], rank=metadata["lora_rank"])
        load_lora_state_dict(generator, torch.load(refined / "lora_state.pt", weights_only=True))
        for name in ("synthesis_weights.pt", "synthesis_noise.pt"):
            if not (refined / name).is_file():
                continue
            incompatible = generator.load_state_dict(
                torch.load(refined / name, weights_only=True), strict=False
            )
            if incompatible.unexpected_keys:
                raise ValueError(f"Unexpected weights in {name}: {incompatible.unexpected_keys}")
        blended = torch.load(refined / "latent_blended.pt", map_location=device, weights_only=True)
        generated = generator.synthesis(blended, noise_mode="const", force_fp32=True)
        base = generated.add(1).div(2).clamp(0, 1)
    del generator, generated
    identity = load_arcface(checkpoint / "arcface.pth", device)
    farl = FaRLTextImageEncoder(checkpoint / "FaRL.pth", device=device)
    bank = load_feature_bank(args.bank)
    perceptual = make_lpips("alex", device)
    with Image.open(refined / "source.png") as source_image:
        reference = np.array(source_image)

    def model_tensor(image):
        return pil_to_tensor(image, 1024).unsqueeze(0).to(device)

    def jpeg_embedding(image, quality):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return extract_arcface(identity, model_tensor(decoded).add(1).div(2))[0]

    source_image = Image.fromarray(reference)
    source_tensor = model_tensor(source_image)
    source_id = extract_arcface(identity, source_tensor.add(1).div(2))[0]
    jpeg_ids = {quality: jpeg_embedding(source_image, quality) for quality in (95, 85)}

    @torch.no_grad()
    def score(image):
        array = np.array(image)
        normalized = model_tensor(image)
        psnr = "Infinity" if np.array_equal(reference, array) else float(
            peak_signal_noise_ratio(reference, array, data_range=255)
        )
        return {
            "mae": float(np.abs(reference.astype(np.int16) - array.astype(np.int16)).mean()),
            "ssim": float(structural_similarity(
                reference, array, data_range=255, channel_axis=-1,
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
            )),
            "psnr_db": psnr,
            "lpips": float(perceptual(source_tensor, normalized).mean()),
            "cosine": float(source_id @ extract_arcface(identity, normalized.add(1).div(2))[0]),
            **{
                f"jpeg{quality}_cosine": float(embedding @ jpeg_embedding(image, quality))
                for quality, embedding in jpeg_ids.items()
            },
        }

    before_image = tensor_to_pil(base * 2 - 1)
    before = score(before_image)
    before_pixels = np.array(before_image, dtype=np.int16)
    candidates = args.job / "candidates"
    candidates.mkdir()
    rows, directions = [], {}
    step_size = settings["watermark_step_px"] / 255.0
    for index, (prompt, epsilon, steps) in enumerate(candidates_settings, 1):
        if prompt not in directions:
            directions[prompt] = attribute_direction(
                bank, farl, prompt, m=settings["candidate_count"], lambda_reg=settings["order_weight"]
            )[0]
        marked = adversarial_watermark(
            base, identity, directions[prompt], epsilon=epsilon, steps=steps, step_size=step_size,
        )
        image = tensor_to_pil(marked.image_01 * 2 - 1)
        row = {"id": f"{index:02}", "prompt": prompt, "epsilon": epsilon, "steps": steps, **score(image)}
        change = np.abs(np.array(image, dtype=np.int16) - before_pixels)
        row.update(watermark_mae=float(change.mean()), watermark_max=int(change.max()))
        if any(not math.isfinite(value) for value in row.values() if isinstance(value, float)):
            raise RuntimeError(f"Non-finite metrics in trial {index}")
        image.save(candidates / f"{row['id']}.png")
        rows.append(row)
        event("sweep", index, total)

    cutoff = settings["jpeg_cutoff"]
    qualified = [row for row in rows if row["jpeg85_cosine"] <= cutoff]
    chosen = max(qualified, key=lambda row: row["ssim"], default=None)
    if chosen:
        shutil.copyfile(candidates / f"{chosen['id']}.png", args.job / "protected.png")
    return {
        "source": "aligned.png",
        "pre_watermark": before,
        "jpeg85_cutoff": cutoff,
        "settings": settings,
        "selected_id": chosen["id"] if chosen else None,
        "trials": rows,
        "reconstruction": metadata["stage_profile"]["reconstruction"],
        "generator_optimization_steps": metadata["generator_optimization_steps"],
        "noise_refinement": metadata.get("noise_refinement_profile", {}).get("refine_synthesis_noise"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    status = args.job / "status.json"
    settings = validate(
        json.loads(status.read_text(encoding="utf-8")).get("settings", {}) if status.is_file() else {}
    )
    started = perf_counter()
    base, refined = args.job / "base", args.job / "refined"
    dependencies = [
        "--stylegan", args.models / "ffhq.pkl", "--arcface", args.models / "arcface.pth",
        "--farl", args.models / "FaRL.pth", "--feature-bank", args.bank, "--device", "cuda",
    ]
    conv_blocks = CONV_BLOCKS[settings["conv_scope"]]
    run_script(
        "inference.py",
        [
            *dependencies, "--e4e", args.models / "e4e.pt", "--image", args.job / "aligned.png",
            "--output-dir", base, "--mode", "combined", "--prompt", settings["prompt"],
            "--profile", "--stop-after", "reconstruction",
            "--tau", settings["tau"], "--candidate-count", settings["candidate_count"],
            "--order-regularization-weight", settings["order_weight"],
            "--lora-blocks", *LORA_BLOCKS[settings["lora_scope"]], "--lora-rank", settings["rank"],
            "--learning-rate", settings["learning_rate"], "--identity-lock-weight", settings["identity_lock"],
            *(["--trainable-conv-blocks", *conv_blocks] if conv_blocks else []),
            "--conv-learning-rate", settings["conv_learning_rate"],
            *(["--cosine-decay"] if settings["cosine_decay"] else []),
            "--reconstruction-l2-weight", settings["l2_weight"],
            "--reconstruction-ssim-weight", settings["ssim_weight"],
            "--lpips-threshold", settings["lpips_threshold"],
            "--max-optimization-steps", settings["reconstruction_steps"],
            "--watermark-epsilon", settings["epsilon"], "--watermark-steps", settings["watermark_steps"],
            "--watermark-step-size", settings["watermark_step_px"] / 255.0,
        ],
        "reconstruct", settings["reconstruction_steps"], re.compile(r"\[generator\] step (\d+)/"),
    )
    if settings["noise_steps"]:
        run_script(
            "refine_combined_noise.py",
            [base, refined, *dependencies, "--steps", settings["noise_steps"],
             "--noise-learning-rate", settings["noise_lr"],
             "--noise-regularization-weight", settings["noise_reg"],
             "--ssim-weight", settings["noise_ssim"]],
            "refine", settings["noise_steps"], re.compile(r"^step=(\d+)/"),
        )
    result = sweep(args, refined if settings["noise_steps"] else base, settings)
    result["seconds"] = round(perf_counter() - started, 2)
    (args.job / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
