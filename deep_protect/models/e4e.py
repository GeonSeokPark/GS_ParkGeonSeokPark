from argparse import Namespace
from pathlib import Path

import torch


class E4EEncoder:
    """Thin wrapper around the bundled e4e/pSp implementation."""

    def __init__(self, checkpoint: Path, device: str = "cuda"):
        from models.e4e.psp import pSp

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"e4e checkpoint not found: {checkpoint}")
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.6
            payload = torch.load(checkpoint, map_location="cpu")
        if "opts" not in payload:
            raise KeyError("e4e checkpoint must contain an 'opts' entry.")
        raw_opts = payload["opts"]
        opts = vars(raw_opts).copy() if hasattr(raw_opts, "__dict__") else dict(raw_opts)
        opts["batch_size"] = 1
        opts["checkpoint_path"] = str(checkpoint)
        opts["device"] = device
        self.model = pSp(Namespace(**opts)).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        # DeepProtect uses the W+ code; the FFHQ ADA generator renders it.
        self.model.encoder.to(device)
        self.device = device

    @torch.no_grad()
    def encode(self, image_01):
        """Return W+ latents for an image tensor in [0, 1]."""

        image_256 = torch.nn.functional.interpolate(
            image_01, size=(256, 256), mode="bilinear", align_corners=False
        )
        image = image_256 * 2.0 - 1.0
        codes = self.model.encoder(image)
        if self.model.opts.start_from_latent_avg:
            codes = codes + self.model.latent_avg
        if codes.ndim != 3 or codes.shape[1:] != (18, 512):
            raise ValueError(f"Expected FFHQ W+ [N, 18, 512], got {tuple(codes.shape)}")
        return codes
