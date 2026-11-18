#!/usr/bin/env python3
"""Run DeepProtect on one aligned face image."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_protect.config import DeepProtectConfig, ModelPaths
from deep_protect.pipeline import DeepProtectPipeline
from deep_protect.utils import seed_everything
from deep_protect.diagnostics import RunProfile


def parse_args():
    parser = argparse.ArgumentParser(description="DeepProtect CVPR 2026 inference")
    parser.add_argument("--image", type=Path, required=True, help="Aligned 1024x1024 RGB face image")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-bank", type=Path, required=True)
    parser.add_argument("--stylegan", type=Path, required=True)
    parser.add_argument("--e4e", type=Path, required=True)
    parser.add_argument("--arcface", type=Path, required=True)
    parser.add_argument("--farl", type=Path, required=True)
    parser.add_argument("--prompt", default="nose")
    parser.add_argument("--mode", choices=("combined", "attribute"), default="combined")
    parser.add_argument("--no-generator-optimization", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tau", type=float, default=0.75)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--order-regularization-weight", type=float, default=1.0)
    parser.add_argument("--lora-blocks", nargs="+", default=DeepProtectConfig.lora_blocks,
                        metavar="BLOCK", help="StyleGAN synthesis block names, such as b16 and b32")
    parser.add_argument("--learning-rate", type=float, default=DeepProtectConfig.learning_rate)
    parser.add_argument("--identity-lock-weight", type=float, default=DeepProtectConfig.identity_lock_weight)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--trainable-conv-blocks", nargs="+", default=(), metavar="BLOCK",
                        help="Also fine-tune convolution/ToRGB weights in these blocks")
    parser.add_argument("--conv-learning-rate", type=float, default=DeepProtectConfig.conv_learning_rate)
    parser.add_argument("--cosine-decay", action="store_true", help="Decay reconstruction learning rates to zero")
    parser.add_argument("--reconstruction-l2-weight", type=float, default=DeepProtectConfig.reconstruction_l2_weight)
    parser.add_argument("--reconstruction-ssim-weight", type=float, default=DeepProtectConfig.reconstruction_ssim_weight)
    parser.add_argument("--max-optimization-steps", type=int, default=450)
    parser.add_argument("--lpips-threshold", type=float, default=0.06)
    parser.add_argument("--watermark-epsilon", type=float, default=0.02)
    parser.add_argument("--watermark-steps", type=int, default=44)
    parser.add_argument("--watermark-step-size", type=float, default=1.0 / 255.0)
    parser.add_argument("--stop-after", choices=("blend", "reconstruction"))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    paths = ModelPaths(args.stylegan, args.e4e, args.arcface, args.farl, args.feature_bank)
    config = DeepProtectConfig(
        tau=args.tau,
        candidate_count=args.candidate_count,
        order_regularization_weight=args.order_regularization_weight,
        lora_blocks=tuple(args.lora_blocks),
        lora_rank=args.lora_rank,
        trainable_conv_blocks=tuple(args.trainable_conv_blocks),
        conv_learning_rate=args.conv_learning_rate,
        cosine_decay=args.cosine_decay,
        reconstruction_l2_weight=args.reconstruction_l2_weight,
        reconstruction_ssim_weight=args.reconstruction_ssim_weight,
        learning_rate=args.learning_rate,
        identity_lock_weight=args.identity_lock_weight,
        max_optimization_steps=args.max_optimization_steps,
        lpips_threshold=args.lpips_threshold,
        watermark_epsilon=args.watermark_epsilon,
        watermark_steps=args.watermark_steps,
        watermark_step_size=args.watermark_step_size,
        device=args.device,
    )
    profile = RunProfile(args.device) if args.profile else None
    if profile:
        with profile.stage("model_load"):
            pipeline = DeepProtectPipeline(paths, config)
    else:
        pipeline = DeepProtectPipeline(paths, config)
    pipeline.protect(
        args.image,
        args.output_dir,
        prompt=args.prompt,
        mode=args.mode,
        optimize=not args.no_generator_optimization,
        verbose=not args.quiet,
        stop_after=args.stop_after,
        profile=profile,
    )
    print(f"Saved DeepProtect artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
