# Third-Party Licenses

LaLa-World itself is released under CC BY-NC 4.0 (see [LICENSE](LICENSE)). It contains or adapts the following
third-party components; their copyright notices and licenses are kept as listed here.

| component | upstream | license | where in this repository |
|---|---|---|---|
| JEPA-WMs | [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms) | CC BY-NC 4.0 | training scaffold (`src/utils/`), simulators, wrappers and CEM planner (`evals/simu_env_planning/`), evaluation protocols (`evals/lala_world/`); files keep their Meta copyright headers |
| RAEv2 | [nanovisionx/RAEv2](https://github.com/nanovisionx/RAEv2) | CC BY-NC 4.0 | `RAEv2/src/stage1`, `RAEv2/src/stage2`, `RAEv2/src/encoders` (trimmed copies); decoder recipe in `decoder/` |
| ViT-MAE (Hugging Face transformers) | [huggingface/transformers](https://github.com/huggingface/transformers) | Apache 2.0 | `RAEv2/src/stage1/decoders/decoder.py` (via RAEv2) |
| Isaac-GR00T | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | Apache 2.0 | `app/lala_policy/flow_head.py` (flow-matching action head, adapted) |
| D4RL | [Farama-Foundation/D4RL](https://github.com/Farama-Foundation/D4RL) | Apache 2.0 | `evals/simu_env_planning/envs/pointmaze_env/`, `evals/lala_world/point_maze_env/` (maze environment, adapted) |
| DINO-WM | [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) | MIT | `evals/simu_env_planning/envs/` (Wall, PointMaze and Push-T wrappers, deformable FlexEnv port) |
| TD-MPC2 | [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) | MIT | `evals/simu_env_planning/envs/wrappers/` |
| Diffusion Policy | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | MIT | `evals/simu_env_planning/envs/pusht_env/` (Push-T environment) |
| transformations.py | Christoph Gohlke | BSD 3-Clause | `evals/simu_env_planning/envs/deformable_env/sim_env/transformations.py` (license in the file header) |
| xArm description and meshes | UFACTORY Inc. | BSD 3-Clause | `evals/simu_env_planning/envs/deformable_env/assets/xarm/` (see its `LICENSE`) |

Apache 2.0 components are modified; a copy of the license is at http://www.apache.org/licenses/LICENSE-2.0.
Simulators installed as dependencies (robosuite, robocasa, Meta-World, LIBERO, MuJoCo) are not included in this
repository and keep their own licenses.

Pretrained models and datasets used by the code (DINOv2, DINOv3, SigLIP2, Qwen3.5, DROID, OpenVid-1M, OmniEdit,
ShareGPT-4o-Image, the LIBERO and RoboCasa Cosmos Policy data) are governed by their own licenses.

## MIT License

Copyright (c) 2025 gaoyuezhou (DINO-WM)
Copyright (c) 2023 Nicklas Hansen (TD-MPC2)
Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab (Diffusion Policy)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
