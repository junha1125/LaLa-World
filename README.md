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
- [Repository layout](#repository-layout)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

> **(October 2026).** This repository is being populated. Environment setup and data preparation are available now.
> Training and evaluation code, pretrained checkpoints and the decoder will follow shortly; the sections marked 🚧 describe what is coming.

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

- [ ] OpenVid change captions (`outputs/captions.jsonl`) will be added to
  [junha1125/openvid-frame-sequences-1M](https://huggingface.co/datasets/junha1125/openvid-frame-sequences-1M);
  until then `data_preparation/openvid/caption_sequences.py` regenerates them with any OpenAI-compatible VLM endpoint.

### Training

> 🚧 **Updating soon.** One trainer (`python -m app.main --fname configs/lala_world/<config>.yaml`), one process per
> GPU, with a config per vision encoder: DINOv2-S/14 (`base.yaml`), DINOv3-L/16 (`dinov3_l.yaml`) and
> SigLIP2-So400M/14 (`siglip2_so400m.yaml`, the LaLa-Policy initialization). 20 epochs x 1M samples at global batch
> 512, about 2 days on 8 x H100.

### Planning evaluation

> 🚧 **Updating soon.** MPC with CEM in latent space on Wall, PointMaze, Push-T, Meta-World (reach, reach-wall),
> Rope and Granular, an offline action score on DROID, and zero-shot reach / place in RoboCasa. The DROID and RoboCasa
> evaluation episodes are the hold-out sets released with [JEPA-WMs](https://github.com/facebookresearch/jepa-wms)
> (`prepare_data.sh eval`). A 1-step probe checks that the prediction beats copying the current frame before planning.

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

> 🚧 **Updating soon.** `python -m app.main --fname configs/lala_policy/<config>.yaml` with a config per benchmark:
> LIBERO (4 suites, agentview + wrist, chunk 16), RoboCasa (24 tasks, about 50 human demos per task) and RoboCasa with
> 300 MimicGen demos per task. `meta.init_from` points at the SigLIP2-So400M/14 LaLa-World checkpoint.

### Evaluation

> 🚧 **Updating soon.** Direct policy rollouts with the Cosmos Policy protocol: LIBERO 4 suites x 10 tasks x 50
> episodes and RoboCasa 24 tasks x 50 episodes on the 5 held-out scenes, run inside the simulator venvs from
> [Installation](#installation).

### Checkpoints

| model | training data | weights |
|---|---|---|
| LaLa-Policy (LIBERO) | LIBERO, 4 suites (Cosmos Policy replays) | coming soon |
| LaLa-Policy (RoboCasa) | RoboCasa, 24 tasks, about 50 human demos per task | coming soon |
| LaLa-Policy (RoboCasa, MimicGen) | RoboCasa, 300 MimicGen demos per task | coming soon |

## Decoder

> 🚧 **Updating soon.** A 12-layer ViT decoder per vision encoder maps patch latents, including the futures LaLa-World
> imagines, back to pixels (L1 + LPIPS, RAEv2's decoder recipe). It is used only for visualization.

## Repository layout

```
data_preparation/   download and preprocessing scripts, normalization specs (specs/, lang_bins/)       available
docker/, scripts/   Dockerfile, dependency and simulator setup, pretrained-model download              available
pyproject.toml      package and extras (sim, data, decoder, dev)                                        available
app/, configs/      LaLa-World and LaLa-Policy training code and configs (entry point app/main.py)     🚧
evals/              planning evaluation, policy rollouts, simulators and the CEM planner                🚧
decoder/            latent-to-pixel decoder                                                             🚧
tests/              unit tests                                                                           🚧
```

## Roadmap

- [x] Environment setup (Docker, pip) and data preparation scripts
- [ ] LaLa-World training and planning evaluation code
- [ ] LaLa-Policy training and evaluation code
- [ ] LaLa-World checkpoints on Hugging Face
- [ ] LaLa-Policy checkpoints on Hugging Face
- [ ] Decoder checkpoints on Hugging Face (one per encoder)
- [ ] OpenVid change captions on the Hub
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
- [nanovisionx/RAEv2](https://github.com/nanovisionx/RAEv2): the decoder recipe.
- [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T): the flow-matching action head of LaLa-Policy.
- [nvidia/LIBERO-Cosmos-Policy](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy) and
  [nvidia/RoboCasa-Cosmos-Policy](https://huggingface.co/datasets/nvidia/RoboCasa-Cosmos-Policy): the policy
  training data and evaluation protocol.

## License

The license will be announced together with the code release. Parts of this repository derive from JEPA-WMs, RAEv2
and Isaac-GR00T; see [Acknowledgements](#acknowledgements).
