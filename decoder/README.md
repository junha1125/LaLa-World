# Decoder

A 12-layer ViT decoder that maps frozen-encoder patch latents, including the latents LaLa-World predicts, back to RGB
images. One decoder per vision encoder (DINOv2-S/14, DINOv3-L/16, SigLIP2-So400M/14). Trained with L1 + LPIPS on
feature-noise-augmented latents (the RAEv2 stage-1 recipe without the GAN); the world model never sees it.

## Train

```bash
pip install lpips
export LALA_DATA_ROOT=/path/to/data
bash decoder/launch.sh decoder/configs/dinov2s.yaml         # 8 GPUs, logs/decoder/dinov2s/{train.log,train.csv,val.jsonl,samples/,checkpoints/}
bash decoder/launch.sh decoder/configs/dinov3l.yaml
bash decoder/launch.sh decoder/configs/siglip2so400m.yaml   # world-model mix + LIBERO / RoboCasa frames (sources_policy.yaml)
```

`NGPU=2 bash decoder/launch.sh ...` runs on fewer GPUs; relaunching with the same config resumes from `checkpoints/latest.pt`.

## Decode

```bash
python -m decoder.recon --ckpt logs/decoder/dinov2s/checkpoints/latest.pt --images a.png b.png --out recon.png   # [originals; reconstructions]
python -m decoder.recon --ckpt logs/decoder/dinov2s/checkpoints/latest.pt --latents preds.pt --out recon.png     # (B, N, D) latents
```

The planning and policy evaluations decode imagined futures with `--decoder-ckpt <checkpoint>` (world model:
`--save-pred-decode N`, policy: `--save-imagine N`); the decoder must match the checkpoint's vision encoder.

- Pretrained decoders, one Hugging Face repo per vision encoder:
  [lala-decoder-dinov2-s](https://huggingface.co/LaLa-World/lala-decoder-dinov2-s),
  [lala-decoder-dinov3-l](https://huggingface.co/LaLa-World/lala-decoder-dinov3-l),
  [lala-decoder-siglip2-so400m](https://huggingface.co/LaLa-World/lala-decoder-siglip2-so400m).
