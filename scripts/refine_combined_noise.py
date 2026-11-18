#!/usr/bin/env python3
"""Refine a combined run using trainable StyleGAN synthesis noise."""

import argparse
import json
from math import isfinite
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageChops
import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from deep_protect.config import resolve_device
from deep_protect.diagnostics import RunProfile
from deep_protect.feature_bank import load_feature_bank
from deep_protect.models.arcface import (
    extract_arcface,
    extract_arcface_with_grad,
    load_arcface,
)
from deep_protect.models.farl import FaRLTextImageEncoder
from deep_protect.models.lora import inject_lora, load_lora_state_dict, lora_state_dict
from deep_protect.models.stylegan import load_stylegan
from deep_protect.optimization import _ssim_loss
from deep_protect.perceptual import lpips_scalar, make_lpips
from deep_protect.utils import pil_to_tensor, seed_everything, tensor_to_pil
from deep_protect.watermark import adversarial_watermark, attribute_direction


def noise_regularizer(noise_states, source):
    loss = source.new_zeros(())
    for buf in noise_states.values():
        grid = buf[None, None]
        while grid.shape[-1] >= 8:
            horizontal = (grid * grid.roll(1, dims=-1)).mean().square()
            vertical = (grid * grid.roll(1, dims=-2)).mean().square()
            loss = loss + horizontal + vertical
            if grid.shape[-1] == 8:
                break
            grid = F.avg_pool2d(grid, 2)
    return loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=Path, help="Combined run with saved LoRA and synthesis weights"
    )
    parser.add_argument("output", type=Path, help="New directory for the refined run")
    for name in ("stylegan", "arcface", "farl", "feature-bank"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--noise-learning-rate", type=float, default=0.06)
    parser.add_argument("--noise-regularization-weight", type=float, default=0.004)
    parser.add_argument("--ssim-weight", type=float, default=1.5)
    parser.add_argument("--prompt")
    parser.add_argument("--watermark-epsilon", type=float)
    parser.add_argument("--watermark-steps", type=int)
    args = parser.parse_args()
    if args.steps < 1 or not (
        isfinite(args.noise_learning_rate) and args.noise_learning_rate > 0
    ):
        parser.error("Refinement steps and learning rate must be positive.")
    if not all(
        isfinite(value) and value >= 0
        for value in (args.noise_regularization_weight, args.ssim_weight)
    ):
        parser.error("Loss weights must be non-negative.")
    if args.watermark_epsilon is not None and not 0 < args.watermark_epsilon <= 1:
        parser.error("Watermark epsilon must be in (0, 1].")
    if args.watermark_steps is not None and args.watermark_steps < 1:
        parser.error("Watermark steps must be positive.")

    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    profile = RunProfile(device)
    seed_everything(42)

    def image_tensor(path):
        with Image.open(path) as image:
            return pil_to_tensor(image, 1024).unsqueeze(0).to(device)

    with profile.stage("load_refinement"):
        generator = load_stylegan(args.stylegan, device)
        inject_lora(
            generator, blocks=metadata["lora_blocks"], rank=metadata["lora_rank"]
        )
        load_lora_state_dict(
            generator, torch.load(args.run / "lora_state.pt", weights_only=True)
        )
        conv_path = args.run / "synthesis_weights.pt"
        conv_state = torch.load(conv_path, weights_only=True) if conv_path.is_file() else {}
        incompatible = generator.load_state_dict(conv_state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(incompatible.unexpected_keys)
        noise_path = args.run / "synthesis_noise.pt"
        if noise_path.is_file():
            loaded = generator.load_state_dict(
                torch.load(noise_path, weights_only=True), strict=False
            )
            if loaded.unexpected_keys:
                raise ValueError(loaded.unexpected_keys)
        latent = torch.load(
            args.run / "latent_blended.pt", map_location=device, weights_only=True
        )
        identity = load_arcface(args.arcface, device)
        lpips = make_lpips("alex", device)
        source_norm = image_tensor(args.run / "source.png")
        source = (source_norm + 1) / 2
        target_id = extract_arcface(
            identity, (image_tensor(args.run / "identity_blended.png") + 1) / 2
        )
        with torch.no_grad():
            reference_render = tensor_to_pil(
                generator.synthesis(latent, noise_mode="const", force_fp32=True)
            )
            with Image.open(args.run / "generator_optimized.png") as previous:
                delta = ImageChops.difference(previous.convert("RGB"), reference_render)
                if any(upper > 1 for _, upper in delta.getextrema()):
                    raise ValueError(
                        "Warm-start weights did not reproduce the saved reconstruction."
                    )
        weights = dict(generator.named_parameters())
        noise_states = {
            name: buf.requires_grad_()
            for name, buf in generator.named_buffers()
            if name.startswith("synthesis.") and name.endswith(".noise_const")
        }
        strengths = {
            name: p.requires_grad_()
            for name, p in weights.items()
            if name.startswith("synthesis.") and name.endswith(".noise_strength")
        }
        lora = [
            p
            for name, p in weights.items()
            if ".down.weight" in name or ".up.weight" in name
        ]
        groups = [
            {"params": lora, "lr": 8e-4},
            {
                "params": [weights[name].requires_grad_() for name in conv_state],
                "lr": 3e-4,
            },
            {"params": list(noise_states.values()), "lr": args.noise_learning_rate},
            {"params": list(strengths.values()), "lr": 5e-4},
        ]
        optimizer = torch.optim.Adam(groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps
        )
        xy = torch.arange(-5, 6, device=device, dtype=torch.float32)
        gaussian = torch.softmax(-xy.square() / (2 * 1.5**2), dim=0)
        kernel = torch.outer(gaussian, gaussian).expand(3, 1, -1, -1).contiguous()
        ref_mean = F.conv2d(source, kernel, groups=3)
        ref_square_mean = F.conv2d(source.square(), kernel, groups=3)

    with profile.stage("refine_synthesis_noise"):
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            generated = generator.synthesis(latent, noise_mode="const", force_fp32=True)
            generated01 = ((generated + 1) / 2).clamp(0, 1)
            l2 = F.mse_loss(generated, source_norm)
            score = lpips_scalar(lpips, generated, source_norm)
            structural_loss = _ssim_loss(
                generated01, source, kernel, ref_mean, ref_square_mean
            )
            current_id = extract_arcface_with_grad(identity, generated01)
            lock = 1 - (target_id * current_id).sum()
            loss = (
                2 * l2
                + 0.6 * score
                + args.ssim_weight * structural_loss
                + metadata["identity_lock_weight"] * lock
                + args.noise_regularization_weight
                * noise_regularizer(noise_states, source)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite reconstruction at step {step}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for buf in noise_states.values():
                    buf.sub_(buf.mean())
                    buf.mul_(buf.square().mean().clamp_min(1e-8).rsqrt())
            if step % 100 == 0 or step == 1 or step == args.steps:
                print(
                    f"step={step}/{args.steps} float_ssim={1 - float(structural_loss.detach()):.5f} "
                    f"lpips={float(score.detach()):.5f} lock={float(lock.detach()):.5f}",
                    flush=True,
                )
        with torch.no_grad():
            optimized = generator.synthesis(latent, noise_mode="const", force_fp32=True)
        base = ((optimized + 1) / 2).clamp(0, 1)
        tensor_to_pil(optimized).save(args.output / "generator_optimized.png")
        torch.save(lora_state_dict(generator), args.output / "lora_state.pt")
        torch.save(
            {name: weights[name].detach().cpu() for name in conv_state},
            args.output / "synthesis_weights.pt",
        )
        torch.save(
            {
                name: value.detach().cpu()
                for name, value in {**noise_states, **strengths}.items()
            },
            args.output / "synthesis_noise.pt",
        )
        for name in (
            "source.png",
            "identity_blended.png",
            "latent_blended.pt",
            "latent_original.pt",
            "blend_candidates.pt",
        ):
            shutil.copyfile(args.run / name, args.output / name)
        del (
            optimizer,
            scheduler,
            generator,
            weights,
            groups,
            lora,
            strengths,
            noise_states,
        )
    with profile.stage("attribute"):
        prompt = args.prompt if args.prompt is not None else metadata["prompt"]
        epsilon = (
            args.watermark_epsilon
            if args.watermark_epsilon is not None
            else metadata["watermark_epsilon"]
        )
        steps = (
            args.watermark_steps
            if args.watermark_steps is not None
            else metadata["watermark_steps"]
        )
        farl = FaRLTextImageEncoder(args.farl, device=device)
        bank = load_feature_bank(args.feature_bank)
        direction, _, _ = attribute_direction(
            bank,
            farl,
            prompt,
            m=metadata["candidate_count"],
            lambda_reg=metadata["order_regularization_weight"],
        )
        mark = adversarial_watermark(
            base,
            identity,
            direction,
            epsilon=epsilon,
            steps=steps,
            step_size=metadata["watermark_step_size"],
        )
        tensor_to_pil(mark.image_01 * 2 - 1).save(args.output / "protected.png")
        torch.save(mark.perturbation.cpu(), args.output / "watermark.pt")
        torch.save(direction.cpu(), args.output / "attribute_direction.pt")
    metadata.update(
        prompt=prompt,
        watermark_epsilon=epsilon,
        watermark_steps=steps,
        attribute_direction_target_sign=mark.target_sign,
        watermark_max_abs=float(mark.perturbation.abs().max()),
        noise_refinement_learning_rate=args.noise_learning_rate,
        noise_refinement_ssim_weight=args.ssim_weight,
        noise_refinement_regularization_weight=args.noise_regularization_weight,
    )
    metadata["warm_start"] = str(args.run)
    metadata["noise_refinement_steps"] = args.steps
    metadata["noise_refinement_profile"] = profile.results
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
