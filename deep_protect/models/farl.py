from pathlib import Path

import torch


class FaRLTextImageEncoder:
    """Load FaRL through the OpenAI CLIP-compatible API."""

    def __init__(self, checkpoint: Path, device: str = "cuda", model_name: str = "ViT-B/16"):
        import clip
        from clip.clip import _transform
        from clip.model import CLIP

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FaRL checkpoint not found: {checkpoint}")
        if model_name != "ViT-B/16":
            raise ValueError("Use the FaRL ViT-B/16 checkpoint.")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(payload, torch.nn.Module):
            model = payload
        else:
            state = payload.get("state_dict", payload)
            state = {key.removeprefix("module."): value for key, value in state.items()}
            for key in ("input_resolution", "context_length", "vocab_size"):
                state.pop(key, None)
            # The official FaRL checkpoint includes a masked-image training head.
            state = {
                key: value for key, value in state.items()
                if key != "visual.mask_token"
                and not key.startswith(("visual.lm_transformer.", "visual.ln_lm.", "visual.lm_head."))
            }
            model = CLIP(
                embed_dim=512, image_resolution=224, vision_layers=12,
                vision_width=768, vision_patch_size=16, context_length=77,
                vocab_size=49408, transformer_width=512, transformer_heads=8,
                transformer_layers=12,
            )
            model.load_state_dict(state, strict=True)
        self.clip = clip
        self.preprocess = _transform(224)
        self.model = model.to(device).eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode_text(self, prompt: str):
        tokens = self.clip.tokenize([prompt]).to(self.device)
        features = self.model.encode_text(tokens).float()
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def encode_images(self, images):
        """Encode a list of PIL images using the FaRL preprocessing pipeline."""

        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        features = self.model.encode_image(batch).float()
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
