<h1 align="center">LaLa-World: Scaling Latent World Models<br>with Natural Language Actions</h1>

<p align="center">
  <a href="https://june-page.github.io/lala-world/"><img alt="Project page" src="https://img.shields.io/badge/Project%20page-lala--world-blue"></a>
  <img alt="arXiv" src="https://img.shields.io/badge/arXiv-coming%20soon-b5212f?logo=arxiv">
  <img alt="Checkpoints" src="https://img.shields.io/badge/🤗%20Checkpoints-coming%20soon-ffc107">
</p>

<p align="center">
  Junha Song<sup>1</sup>, Geonmo Gu<sup>1</sup>, Jeongeun Park<sup>2</sup>, Heejeong Nam<sup>3</sup>, Lucas Maes<sup>4</sup>,<br>
  Byeongho Heo<sup>1</sup>, Dongyoon Han<sup>1</sup>, Sangdoo Yun<sup>1</sup>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/naver_ai_lab_dark.png">
    <img src="assets/naver_ai_lab_light.png" alt="NAVER AI Lab" width="190">
  </picture>
</p>

<p align="center">
  <sup>1</sup>NAVER AI Lab &nbsp; <sup>2</sup>Georgia Institute of Technology &nbsp; <sup>3</sup>Brown University &nbsp; <sup>4</sup>Mila &amp; Université de Montréal
</p>

**LaLa-World** is a **La**nguage-conditioned **La**tent **World** model that uses natural language as a shared action
interface to learn from simulation, robotics, images, and videos. It predicts future visual representations for
efficient planning across domains and can be fine-tuned into **LaLa-Policy**, which jointly predicts robot actions and
future representations.

- Higher average MPC planning performance than latent world models trained separately for each domain or task; adding
  image and video data further improves planning and robustness to action phrasing.
- Planning performance comparable to a matched RAEv2-based video generation model with 6× less training cost and
  13.9× faster planning.
- LaLa-Policy is competitive with Cosmos Policy on LIBERO and RoboCasa with up to 5.5× faster inference.

<!-- teaser: add assets/teaser.png here -->

## Contents

- [Installation](#installation)
- [LaLa-World](#lala-world): [Data](#data) · [Training](#training) · [Planning evaluation](#planning-evaluation) · [Checkpoints](#checkpoints)
- [LaLa-Policy](#lala-policy): [Data](#data-1) · [Training](#training-1) · [Evaluation](#evaluation) · [Checkpoints](#checkpoints-1)
- [Decoder](#decoder)
- [Video generation baseline](#video-generation-baseline)
- [Repository layout](#repository-layout)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

> **(October 2026).** This repository is being populated. Environment setup, data preparation, the training and
> evaluation code and the decoder are available now; the pretrained LaLa-World and LaLa-Policy checkpoints will follow shortly.

## Installation

Tested on Ubuntu 24.04 with 8 x H100, Python 3.12, PyTorch 2.8 (`nvcr.io/nvidia/pytorch:25.06-py3` and a plain
PyPI install), transformers 5.19, flash-linear-attention 0.5.2.

**Docker (recommended)**

```bash
git clone https://github.com/junha1125/LaLa-World.git && cd LaLa-World
docker build -t lala-world -f docker/Dockerfile .
docker run --gpus all -it --shm-size 128g -v $PWD:/workspace/LaLa-World -v /path/to/data:/workspace/LaLa-World/data lala-world
```

**pip**

```bash
conda create -n lala-world python=3.12 -y && conda activate lala-world
pip install "torch>=2.8" torchvision
bash scripts/install_deps.sh        # apt (ffmpeg, EGL/OSMesa) + pip install -e ".[sim,data,decoder,dev]"
```

**Pretrained encoders and LLM** (DINOv3 is gated: `hf auth login` first)

```bash
bash scripts/download_models.sh     # hf-models/: Qwen3.5-0.8B, DINOv2 ViT-S/14, DINOv3 ViT-L/16, SigLIP2 So400M/14
```

**Simulators for planning evaluation.** Wall, PointMaze, Push-T and Meta-World come with the `sim` extra
(`export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` for Meta-World). Rope and Granular need the PyFleX build
from [DINO-WM](https://github.com/apple/ml-dino-wm); point `PYFLEX_HOME` at it.

**Simulators for policy evaluation.** LIBERO and RoboCasa pin their own `robosuite`/`mujoco`, so the evaluation
drivers use two small venvs that share the main site-packages:

```bash
bash scripts/install_sim_envs.sh libero      # third_party/venv-libero
bash scripts/install_sim_envs.sh robocasa    # third_party/venv-robocasa (+ ~20 GB kitchen assets)
```

**Data root.** All datasets are downloaded and preprocessed under `data/` (or `$LALA_DATA_ROOT`) by
`data_preparation/prepare_data.sh`; the per-model commands are in the sections below, and per-dataset commands,
sizes and the output layout are in [`data_preparation/README.md`](data_preparation/README.md).

## LaLa-World

The world model: a frozen vision encoder, a Qwen3.5-0.8B predictor (24 layers) and a text interface for actions.
Numerical actions are rendered into text with the normalization specs in `data_preparation/specs/` and
`data_preparation/lang_bins/`, and the same predictor consumes the change captions and edit instructions of the image
and video data. Planning is MPC with CEM in latent space: candidate action chunks are written in the training text
format, rolled out by the predictor and scored against the encoded goal. Setup: [Installation](#installation).

### Data

```bash
bash data_preparation/prepare_data.sh sim      --root data   # Wall, PointMaze, Push-T, Meta-World, Rope, Granular
bash data_preparation/prepare_data.sh droid    --root data   # DROID (LeRobot release, one camera, 165 GB)
bash data_preparation/prepare_data.sh openvid  --root data   # OpenVid-1M frame sequences + change captions
bash data_preparation/prepare_data.sh edit     --root data   # OmniEdit (25% shards) + ShareGPT-4o-Image
bash data_preparation/prepare_data.sh eval     --root data   # planning-only data: RoboCasa demos, Franka trajectories (5 GB)
bash data_preparation/prepare_data.sh world    --root data   # all of the above
```

Outputs: `data/jsonl/traj/<dataset>.jsonl` (one line per episode, frames under `data/<dataset>/frames/`) and
`data/jsonl/pair/{omniedit,sharegpt}.jsonl`. The action-text normalization specs used by both the builders and
the planner are shipped in `data_preparation/specs/` and `data_preparation/lang_bins/`.

The OpenVid frame sequences and change captions are on the Hub:
[LaLa-World/openvid-frame-sequences-1M](https://huggingface.co/datasets/LaLa-World/openvid-frame-sequences-1M)
(`prepare_data.sh openvid` downloads both).

### Training

```bash
python -m app.main --fname configs/lala_world/base.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
```

| config | vision encoder | predictor | context | data |
|---|---|---|---|---|
| `base.yaml` | DINOv2-S/14 | Qwen3.5-0.8B, 24 layers | up to 3 frames | 7 control datasets (DROID capped at 1.5M) + OpenVid + OmniEdit + ShareGPT |
| `dinov3_l.yaml` | DINOv3-L/16 | Qwen3.5-0.8B, 24 layers | up to 3 frames | same, full DROID |
| `siglip2_so400m.yaml` | SigLIP2-So400M/14 | Qwen3.5-0.8B, 24 layers | 1 frame | same; the LaLa-Policy initialization |

One process per GPU, 20 epochs x 1M samples at global batch 512 (`batch_size x #GPUs x grad_accum_steps`), about
2 days on 8 x H100. Outputs land in `folder/` (`train.csv`, `val.jsonl`, `lala-e<epoch>.pth.tar`); `meta.quick_debug: true`
turns any config into a 4-step smoke run. Validation logs the prediction loss per source group, the same loss with
the action text blanked, and `rel_l2` (error relative to copying the last frame).

### Planning evaluation

```bash
CK=checkpoints/lala-world-dinov2-s.pth.tar
python -m evals.lala_world.probe_onestep --ckpt $CK --env wall pusht --val-only --ctxt-windows 1 2 3   # 1-step check: the prediction must beat copying the current frame
python -m evals.lala_world.run_planning --ckpt $CK --env wall --episodes 96                             # wall | point_maze | pusht
python -m evals.lala_world.run_planning --ckpt $CK --env metaworld --mw-task reach --goal-H 20 --horizon 6 --iterations 15 --num-act-stepped 3 --episodes 48
python -m evals.lala_world.run_planning --ckpt $CK --env rope --goal-H 5 --horizon 5 --iterations 30 --num-elites 30 --num-act-stepped 1 --episodes 24   # rope | granular (PyFleX)
python -m evals.lala_world.run_planning_droid_franka --ckpt $CK --horizon 3 --ctxt-window 2 --episodes 64   # offline action score on DROID
bash evals/lala_world/driver_robocasa.sh $CK reach 0 plan 32                                            # zero-shot RoboCasa reach | place (RoboCasa venv)
```

MPC with CEM in latent space on Wall, PointMaze, Push-T, Meta-World (reach, reach-wall), Rope and Granular, an
offline action score on DROID, and zero-shot reach / place in RoboCasa; results go to `logs/planning/<env>/<tag>/result.json`.
The DROID and RoboCasa episodes are the hold-out sets released with [JEPA-WMs](https://github.com/facebookresearch/jepa-wms)
(`prepare_data.sh eval`). Per-benchmark settings, sharding over GPUs and decoding the imagined futures:
[`evals/lala_world/README.md`](evals/lala_world/README.md).

### Checkpoints

| model | vision encoder | predictor | weights |
|---|---|---|---|
| LaLa-World (DINOv2-S/14) | DINOv2 ViT-S/14 | Qwen3.5-0.8B, 24 layers | coming soon |
| LaLa-World (DINOv3-L/16) | DINOv3 ViT-L/16 | Qwen3.5-0.8B, 24 layers | coming soon |
| LaLa-World (SigLIP2-So400M/14) | SigLIP2 So400M/14 | Qwen3.5-0.8B, 24 layers | coming soon |

## LaLa-Policy

The policy: the pretrained LaLa-World (SigLIP2-So400M/14 encoder, 24-layer Qwen3.5 predictor) fine-tuned to jointly
predict the action chunk and the future visual representation. Actions come from a flow-matching action head (the
DiT of [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)); the world-model pretraining is the initialization, with
language-model-only and random initialization as controls. It is evaluated with the Cosmos Policy protocol on LIBERO
and RoboCasa. Setup: [Installation](#installation).

### Data

```bash
bash data_preparation/prepare_data.sh libero       --root data   # nvidia/LIBERO-Cosmos-Policy (22 GB)
bash data_preparation/prepare_data.sh robocasa     --root data   # nvidia/RoboCasa-Cosmos-Policy (152 GB)
bash data_preparation/prepare_data.sh robocasa_mg  --root data   # RoboCasa MimicGen, 300-demo protocol (816 GB)
bash data_preparation/prepare_data.sh policy       --root data   # all of the above
```

Outputs: `data/cosmos_libero/`, `data/cosmos_robocasa/`, `data/robocasa_mg/` (episode JSONL + `stats*.json`;
images stay in the HDF5 files).

### Training

```bash
python -m app.main --fname configs/lala_policy/base.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
```

| config | data | views | chunk | epochs |
|---|---|---|---|---|
| `base.yaml` | LIBERO, 4 suites (Cosmos Policy replays) | agentview + wrist | 16 | 8 |
| `robocasa.yaml` | RoboCasa, 24 tasks, about 50 human demos per task | left + right + wrist | 32 | 12 |
| `robocasa_mg300.yaml` | RoboCasa, 300 MimicGen demos per task | left + right + wrist | 32 | 3 |

`meta.init_from` is the SigLIP2-So400M/14 LaLa-World checkpoint (`null` = no world-model pretraining;
`model.llm.pretrained: false` on top = random initialization). Global batch 256; about 5 hours for LIBERO and
14 hours for RoboCasa on 8 x H100.

### Evaluation

```bash
bash evals/lala_policy/run_libero_all.sh checkpoints/lala-policy-libero.pth.tar lala-policy-libero 50 16          # 4 suites x 10 tasks x 50 episodes, 16 processes
bash evals/lala_policy/run_robocasa_all.sh checkpoints/lala-policy-robocasa50.pth.tar lala-policy-robocasa 50 12  # 24 tasks x 50 episodes on the 5 held-out scenes
```

Direct policy rollouts with the Cosmos Policy protocol, one process per task inside the simulator venvs from
[Installation](#installation); each script ends with `evals.lala_policy.collect` (`logs/policy_eval/summary.md`,
Wilson 95% CIs). Single tasks and decoding the imagined frames:
[`evals/lala_policy/README.md`](evals/lala_policy/README.md).

### Checkpoints

| model | training data | weights |
|---|---|---|
| LaLa-Policy (LIBERO) | LIBERO, 4 suites (Cosmos Policy replays) | coming soon |
| LaLa-Policy (RoboCasa) | RoboCasa, 24 tasks, about 50 human demos per task | coming soon |
| LaLa-Policy (RoboCasa, MimicGen) | RoboCasa, 300 MimicGen demos per task | coming soon |

## Decoder

A 12-layer ViT decoder per vision encoder maps patch latents, including the futures LaLa-World imagines, back to
pixels (L1 + LPIPS, RAEv2's decoder recipe). It is used only for visualization. Pretrained decoders:
[LaLa-World/lala-decoder](https://huggingface.co/LaLa-World/lala-decoder).

```bash
bash decoder/launch.sh decoder/configs/dinov2s.yaml                                                        # train; dinov3l.yaml, siglip2so400m.yaml
python -m decoder.recon --ckpt checkpoints/lala-decoder-dinov2-s.pt --images a.png b.png --out recon.png   # [originals; reconstructions]
```

The planning and policy evaluations decode imagined futures with `--decoder-ckpt` (`--save-pred-decode N` and
`--save-imagine N`); details in [`decoder/README.md`](decoder/README.md).

## Video generation baseline

The paper compares LaLa-World with a generative world model built on [RAEv2](https://github.com/nanovisionx/RAEv2):
the same DINOv3-L/16 encoder, language actions and training transitions, but a DiT-XL predictor that denoises the
next-frame latent instead of regressing it. `RAEv2/` holds the trimmed upstream code, the world-model trainer and the
planning backend; setup and details in [`RAEv2/README.md`](RAEv2/README.md).

```bash
cd RAEv2
pip install timm && pip install --no-deps "git+https://github.com/nanovisionx/gmuon.git"
bash scripts/download_weights.sh                      # RAE encoder/decoder, ImageNet stage-2 DiT, Qwen3-0.6B
bash scripts/train.sh configs/rae_xl.yaml             # 8 x H100, 12 h  (configs/rae_dit7.yaml: 1/4-depth DiT)
cd ..
CK=RAEv2/runs/rae_xl/checkpoints/step-0073767.pt
bash RAEv2/scripts/eval_planning.sh $CK RAEv2/runs/rae_xl/planning                                # 6 domains, 8 GPUs, 1 denoising step (NFE=5 for 5)
python -m evals.lala_world.run_planning --wm rae --ckpt $CK --env wall --episodes 96 --rae-nfe 1   # one domain
```

## Repository layout

```
data_preparation/   download and preprocessing scripts, normalization specs (specs/, lang_bins/)
docker/, scripts/   Dockerfile, dependency and simulator setup, pretrained-model download
pyproject.toml      package and extras (sim, data, decoder, dev)
app/, configs/      LaLa-World and LaLa-Policy training code and configs (entry point app/main.py)
evals/              planning evaluation, policy rollouts, simulators and the CEM planner
decoder/            latent-to-pixel decoder
RAEv2/              RAEv2 generative world-model baseline (trainer and planning backend)
tests/              unit tests (pytest tests)
```

## Roadmap

- [ ] LaLa-World checkpoints on Hugging Face
- [ ] LaLa-Policy checkpoints on Hugging Face
- [ ] Planning evaluation on the environments provided by [Stable-WorldModel](https://github.com/galilai-group/stable-worldmodel)

## Citation

```bibtex
@article{song2026lalaworld,
  title   = {LaLa-World: Scaling Latent World Models with Natural Language Actions},
  author  = {Song, Junha and Gu, Geonmo and Park, Jeongeun and Nam, Heejeong and Maes, Lucas and Heo, Byeongho and Han, Dongyoon and Yun, Sangdoo},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgements

- [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms): this repository is built on it (training
  scaffold, simulators, planner, evaluation data).
- [nanovisionx/RAEv2](https://github.com/nanovisionx/RAEv2): the generative world-model baseline and the decoder recipe.
- [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T): the flow-matching action head of LaLa-Policy.
- [nvidia/LIBERO-Cosmos-Policy](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy) and
  [nvidia/RoboCasa-Cosmos-Policy](https://huggingface.co/datasets/nvidia/RoboCasa-Cosmos-Policy): the policy
  training data and evaluation protocol.

## License

This project is licensed under CC-BY-NC 4.0. See [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md) for third-party components.
