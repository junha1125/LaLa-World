"""Reconstruct pixels from a trained decoder checkpoint.

  python -m decoder.recon --ckpt <ckpt> --images a.png b.png --out recon.png   # encode -> decode -> [orig; recon]
  python -m decoder.recon --ckpt <ckpt> --latents preds.pt --out recon.png     # decode saved (B, N, D) latents
--noise N adds sigma = N x feature RMS noise before decoding (robustness at the corruption level of predictions).
"""

import argparse

import numpy as np
import torch
from PIL import Image

from decoder.encoders import build_frozen_encoder, normalize_pixels
from decoder.model import load_decoder


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images", nargs="*", default=None)
    p.add_argument("--latents", default=None, help=".pt tensor (B, N, D) or (N, D)")
    p.add_argument("--out", required=True)
    p.add_argument("--noise", type=float, default=0.0, help="sigma in units of the stored feature RMS")
    p.add_argument("--weights", default="ema", choices=["ema", "decoder"])
    args = p.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dec = load_decoder(args.ckpt, device, args.weights)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    rows = []
    if args.images:
        enc = build_frozen_encoder(ck["vision"], device)
        s = enc.image_size
        x01 = (
            torch.stack(
                [
                    torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((s, s), Image.BILINEAR))).permute(
                        2, 0, 1
                    )
                    for f in args.images
                ]
            )
            .float()
            .div(255)
            .to(device)
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = enc(normalize_pixels(x01, enc))["patches"].float()
        rows.append(np.concatenate((x01.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8), axis=1))
    else:
        z = torch.load(args.latents, map_location=device).float()
        z = z.unsqueeze(0) if z.dim() == 2 else z
    if args.noise > 0:
        z = z + args.noise * ck["feat_rms"] * torch.randn_like(z)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        rec = dec(z).float().clamp(0, 1)
    rows.append(np.concatenate((rec.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8), axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"saved {args.out} ({'orig; ' if args.images else ''}recon, {rec.shape[0]} column(s))")


if __name__ == "__main__":
    main()
