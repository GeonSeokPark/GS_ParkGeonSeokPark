#!/usr/bin/env python3
"""Require an actual CUDA allocation, forward pass, and backward pass."""
import json
import platform
import sys

import torch
import torchvision


def main():
    print(json.dumps({
        "platform": platform.platform(), "python": sys.version.split()[0],
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
    }), flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot access CUDA.")
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    x = torch.ones(64, 64, device="cuda", requires_grad=True)
    (x @ x).mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError("CUDA gradient is non-finite.")
    print(json.dumps({
        "gpu": props.name, "compute_capability": [props.major, props.minor],
        "free_mib": free // 2**20, "total_mib": total // 2**20, "cuda_backward": "ok",
    }))


if __name__ == "__main__":
    main()
