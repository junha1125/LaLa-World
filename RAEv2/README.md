# RAEv2 generative world model (baseline)

The video-generation baseline of the paper: a frozen [RAEv2](https://github.com/nanovisionx/RAEv2) autoencoder
(DINOv3-L/16 encoder + ViT-XL decoder), a DiT-XL predictor (0.88B, initialised from the public ImageNet stage-2
checkpoint) and a frozen Qwen3-0.6B text encoder. It is trained with flow matching on the same transitions and the
same language actions as LaLa-World, `(x_t, z) -> x_{t+1}`, and planned with the same CEM stack. `src/stage1`,
`src/stage2`, `src/encoders` are trimmed copies of the upstream code; `src/wm` is the world-model part.

## Setup

On top of the LaLa-World environment:

```bash
pip install timm
pip install --no-deps "git+https://github.com/nanovisionx/gmuon.git"      # GMuon optimizer (import gram_newton_schulz)
bash scripts/download_weights.sh                                           # RAE, stage-2 DiT, Qwen3-0.6B -> pretrained_models/
```

The DINOv3 code is fetched with `torch.hub` from `facebookresearch/dinov3`; for an offline machine clone it and set
`DINOV3_REPO_DIR=/path/to/dinov3` (or `dinov3_repo_dir` in the config).

`PYTHONPATH=src pytest tests` runs the CPU unit tests of `src/wm`.

## Data

The configs read the LaLa-World JSONL files (`jsonl/traj/*.jsonl`, `jsonl/pair/*.jsonl`) under `data.root`
(`LALA_DATA_ROOT` or `LALA_DATA` overrides it). Every trajectory source is a single transition (`clip_len: 2`); windows, val split and
text rendering are the ones of the LaLa-World loaders.

## Training

```bash
cd RAEv2
bash scripts/train.sh configs/rae_xl.yaml        # DiT-XL, 8 x H100, stops at 12 h or 80k steps (global batch 256)
bash scripts/train.sh configs/rae_dit7.yaml      # 1/4-depth DiT, same number of steps as the rae_xl run
```

`runs/<name>/checkpoints/step-XXXXXXX.pt` are the snapshots (bf16 model + EMA + config + training wall-clock),
`latest.pt` also holds the optimizer. Re-running the same command resumes from `latest.pt`;
`--resume-snapshot <step-XXXXXXX.pt>` continues from a snapshot with a fresh optimizer. `train.csv` has the loss and
`rel_l2` (one-shot prediction error relative to copying the current frame), `val.jsonl` the per-source validation
(`rel_l2`, in-batch retrieval R@1, flow-matching loss) every 5k steps.

The 12-hour run reaches step 73,767 on 8 H100s (~430 samples/s with `torch.compile`). `global_batch_size: 256` is 32 per
GPU (~75 GB); with fewer GPUs lower it (`NGPU=2` needs 64) and expect a different run.

## Evaluation (planning)

A snapshot is planned with the LaLa-World planning scripts and `--wm rae`; episodes, goals, CEM and the success
rules are shared, only the world model changes. Per predicted frame the DiT runs `--rae-nfe` Euler steps from pure
noise (1 = one-shot x-prediction, 5 in the paper's second setting) with text classifier-free guidance `--rae-cfg`
(2.0); the CEM cost is the L2 distance in the RAE latent space (`--rae-cost latent`; `pixel` decodes and compares images).

```bash
cd ..   # LaLa-World root
export LALA_DATA_ROOT=/path/to/data; CK=RAEv2/runs/rae_xl/checkpoints/step-0073767.pt
python -m evals.lala_world.run_planning --wm rae --ckpt $CK --env wall --episodes 96 --rae-nfe 1 --rae-cfg 2.0 --rae-cost latent --out RAEv2/runs/rae_xl/planning
python -m evals.lala_world.run_planning_droid_franka --wm rae --ckpt $CK --horizon 3 --ctxt-window 1 --episodes 64 --rae-nfe 1 --out RAEv2/runs/rae_xl/planning
```

The six domains of the paper on 8 GPUs (episode-sharded, then merged into `summary.md`):

```bash
bash RAEv2/scripts/eval_planning.sh RAEv2/runs/rae_xl/checkpoints/step-0073767.pt RAEv2/runs/rae_xl/planning
NFE=5 bash RAEv2/scripts/eval_planning.sh RAEv2/runs/rae_xl/checkpoints/step-0073767.pt RAEv2/runs/rae_xl/planning-nfe5
```

Our 12-hour snapshot (step 73,767) with one denoising step, full protocol, one seed:

| | Wall | PointMaze | Push-T | MetaWorld-R | MetaWorld-RW | DROID |
|---|---|---|---|---|---|---|
| episodes | 96 | 24 | 96 | 24 | 24 | 64 |
| success % / action score | 64.6 | 50.0 | 10.4 | 79.2 | 45.8 | 26.0 |
| planning s / episode (1 H100) | 618 | 639 | 624 | 1972 | 1972 | 163 |

Five denoising steps cost ~4.8x the planning time, the 1/4-depth DiT ~0.37x.

## Notes

- cuDNN SDPA is disabled in `train.py`: its bf16 backward produced NaN gradients mid-run. Non-finite steps are skipped
  (training aborts after 20 in a row) and a snapshot is never written from non-finite weights.
- Gradients are all-reduced manually instead of through DDP (DDP + `torch.compile` crashed on the NGC PyTorch image).
- The paper run also included ImageNet relation-augmented pairs in the training mix, which the public data scripts do not build.
