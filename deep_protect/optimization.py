from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .models.arcface import extract_arcface, extract_arcface_with_grad
from .models.lora import inject_lora, lora_state_dict
from .perceptual import lpips_scalar, make_lpips


@dataclass
class OptimizationResult:
    generator: torch.nn.Module
    lora_state: Dict[str, torch.Tensor]
    initial_image: torch.Tensor
    optimized_image: torch.Tensor
    steps: int
    final_l2: float
    final_lpips: float
    final_identity_lock: float
    injected_modules: list
    conv_state: Dict[str, torch.Tensor]


def _ssim_loss(image, reference, kernel, reference_mean, reference_square_mean):
    # Valid Gaussian window, per RGB channel, with data range 1.
    mean = F.conv2d(image, kernel, groups=3)
    square_mean = F.conv2d(image.square(), kernel, groups=3)
    cross_mean = F.conv2d(image * reference, kernel, groups=3)
    variance = (square_mean - mean.square()).clamp_min(0)
    reference_variance = (reference_square_mean - reference_mean.square()).clamp_min(0)
    covariance = cross_mean - mean * reference_mean
    luminance = (2 * mean * reference_mean + 0.01**2) / (mean.square() + reference_mean.square() + 0.01**2)
    structure = (2 * covariance + 0.03**2) / (variance + reference_variance + 0.03**2)
    return 1 - (luminance * structure).mean()


def optimize_generator(
    original_generator,
    blended_w: torch.Tensor,
    source_image_01: torch.Tensor,
    identity_model,
    *,
    blocks=("b16", "b32"),
    rank: int = 8,
    lora_scale: float = 1.0,
    learning_rate: float = 3e-4,
    identity_lock_weight: float = 0.1,
    trainable_conv_blocks=(),
    conv_learning_rate: float = 1e-4,
    cosine_decay: bool = False,
    l2_weight: float = 1.0,
    ssim_weight: float = 0.0,
    max_steps: int = 450,
    lpips_threshold: float = 0.06,
    lpips_network: str = "alex",
    device: str = "cuda",
    verbose: bool = True,
) -> OptimizationResult:
    """Optimize Eq. (4) with partial StyleGAN LoRA tuning."""

    tuned = original_generator
    injected = inject_lora(tuned, blocks=blocks, rank=rank, scale=lora_scale)
    tuned.eval()
    lora_parameters = [parameter for parameter in tuned.parameters() if parameter.requires_grad]
    conv_parameters = {}
    for name in dict.fromkeys(trainable_conv_blocks):
        block = getattr(tuned.synthesis, name, None)
        if block is None:
            raise ValueError(f"StyleGAN generator has no synthesis block {name!r}.")
        for layer_name in ("conv0", "conv1", "torgb"):
            layer = getattr(block, layer_name, None)
            if layer is not None:
                conv_parameters[f"synthesis.{name}.{layer_name}.weight"] = layer.weight.requires_grad_(True)

    source = source_image_01.to(device).clamp(0, 1)
    source_norm = source * 2.0 - 1.0
    with torch.no_grad():
        initial = tuned.synthesis(blended_w.to(device), noise_mode="const", force_fp32=True).detach()
        initial_norm = initial
        target_identity = extract_arcface(identity_model, (initial_norm + 1.0) / 2.0).detach()

    lpips_model = make_lpips(lpips_network, device)
    parameter_groups = [{"params": lora_parameters}]
    if conv_parameters:
        parameter_groups.append({"params": list(conv_parameters.values()), "lr": conv_learning_rate})
    optimizer = torch.optim.Adam(parameter_groups, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps) if cosine_decay else None
    if ssim_weight:
        coordinates = torch.arange(-5, 6, device=source.device, dtype=source.dtype)
        gaussian = torch.exp(-coordinates.square() / (2 * 1.5**2))
        gaussian = gaussian / gaussian.sum()
        kernel = torch.outer(gaussian, gaussian).expand(3, 1, -1, -1).contiguous()
        reference_mean = F.conv2d(source, kernel, groups=3)
        reference_square_mean = F.conv2d(source.square(), kernel, groups=3)

    final_values = (float("inf"), float("inf"), float("inf"))
    completed = 0
    for step in range(max_steps):
        generated = tuned.synthesis(blended_w.to(device), noise_mode="const", force_fp32=True)
        l2 = F.mse_loss(generated, source_norm)
        perceptual = lpips_scalar(lpips_model, generated, source_norm)
        generated_identity = extract_arcface_with_grad(identity_model, (generated + 1.0) / 2.0)
        identity_lock = 1.0 - (target_identity * generated_identity).sum(dim=1).mean()
        loss = l2_weight * l2 + perceptual + identity_lock_weight * identity_lock
        if ssim_weight:
            loss = loss + ssim_weight * _ssim_loss(
                ((generated + 1) / 2).clamp(0, 1), source, kernel,
                reference_mean, reference_square_mean,
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        completed = step + 1
        final_values = (float(l2.detach()), float(perceptual.detach()), float(identity_lock.detach()))
        if verbose and (step == 0 or completed % 25 == 0):
            print(
                f"[generator] step {completed:03d}/{max_steps} "
                f"loss={float(loss.detach()):.5f} l2={final_values[0]:.5f} "
                f"lpips={final_values[1]:.5f} id_lock={final_values[2]:.5f}"
            )
        if final_values[1] <= lpips_threshold:
            break

    with torch.no_grad():
        optimized = tuned.synthesis(blended_w.to(device), noise_mode="const", force_fp32=True).detach()
    return OptimizationResult(
        tuned,
        lora_state_dict(tuned),
        initial.detach(),
        optimized,
        completed,
        final_values[0],
        final_values[1],
        final_values[2],
        injected,
        {name: parameter.detach().cpu() for name, parameter in conv_parameters.items()},
    )
