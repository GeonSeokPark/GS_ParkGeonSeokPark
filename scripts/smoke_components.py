#!/usr/bin/env python3
"""Exercise individual operations with real weights and deterministic tensor fixtures."""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from deep_protect.blending import identity_blend
from deep_protect.diagnostics import RunProfile
from deep_protect.feature_bank import FeatureBank, load_feature_bank, save_feature_bank
from deep_protect.models.arcface import load_arcface
from deep_protect.models.e4e import E4EEncoder
from deep_protect.models.farl import FaRLTextImageEncoder
from deep_protect.models.stylegan import load_stylegan
from deep_protect.optimization import optimize_generator
from deep_protect.utils import seed_everything
from deep_protect.watermark import adversarial_watermark, attribute_direction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stylegan", "e4e", "arcface", "farl"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = args.device
    seed_everything(42)
    profile = RunProfile(device)
    source = torch.full((1, 3, 1024, 1024), 0.5, device=device)
    fixture = FeatureBank(torch.randn(60, 18, 512), torch.randn(60, 512), torch.randn(60, 512),
                          [f"fixture_{i:02d}/인물.png" for i in range(60)])
    with profile.stage("bank_roundtrip"):
        with TemporaryDirectory() as directory:
            for suffix in ("h5", "npz"):
                path = Path(directory) / f"fixture.{suffix}"
                save_feature_bank(path, fixture)
                loaded = load_feature_bank(path)
                assert loaded.names == fixture.names
                for name in ("latents", "clip_features", "id_features"):
                    torch.testing.assert_close(getattr(loaded, name), getattr(fixture, name))
    bank = fixture.to(device)
    with profile.stage("load_models"):
        e4e = E4EEncoder(args.e4e, device)
        farl = FaRLTextImageEncoder(args.farl, device)
        identity = load_arcface(args.arcface, device)
        generator = load_stylegan(args.stylegan, "cpu")
    with profile.stage("blend_fixture"):
        from PIL import Image
        w = e4e.encode(source)
        clip = farl.encode_images([Image.new("RGB", (1024, 1024), (128, 128, 128))])
        blended = identity_blend(w, clip, bank)
        styles = [0, 1, *range(7, 18)]
        torch.testing.assert_close(blended.latent[:, styles], w[:, styles], rtol=0, atol=0)
        for index in range(2, 7):
            options = bank.latents[blended.candidate_indices, index]
            assert (options == blended.latent[0, index]).all(dim=1).any()
    with profile.stage("reconstruction_one_step_fixture"):
        generator.to(device)
        result = optimize_generator(generator, w, source, identity, device=device, max_steps=1)
        assert result.steps == 1 and result.lora_state
        assert result.optimized_image.shape == source.shape
        assert torch.isfinite(result.optimized_image).all()
        del result, generator
    with profile.stage("attribute_44_steps_fixture"):
        direction, lda, _ = attribute_direction(bank, farl, "nose")
        assert len(set(lda.selected_indices_)) == 60
        torch.testing.assert_close(direction.norm(), torch.ones((), device=device))
        output = adversarial_watermark(source, identity, direction)
        assert output.steps == 44 and torch.isfinite(output.image_01).all()
        assert 0 < (output.image_01 - source).abs().max() <= 0.020001


if __name__ == "__main__":
    main()

