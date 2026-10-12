# Data preparation

All scripts take `--root DIR` (default `$LALA_DATA_ROOT`, else `./data`) and write into that directory.
`prepare_data.sh <target>` runs the download and build steps below in order; each step can also be run on its own.

```bash
bash data_preparation/prepare_data.sh world  --root data --workers 24    # sim + droid + openvid + edit + eval
bash data_preparation/prepare_data.sh policy --root data --workers 24    # libero + robocasa + robocasa_mg
```

JSONL files embed absolute paths, so build the data where it will stay (or rebuild after moving it).

## Layout

```
data/
  sim_raw/            raw downloads: pusht_noise/ point_maze/ wall_single/ metaworld/data/ deformable/{rope,granular}/ droid_lerobot/
  <dataset>/frames/   extracted JPEG frames (wall, point_maze, pusht, metaworld, rope, granular, droid)
  jsonl/traj/         <dataset>.jsonl   one line per episode (world-model training)
  jsonl/pair/         omniedit.jsonl sharegpt.jsonl   one line per (image, instruction, image) pair
  openvid_raw/        sequences.jsonl frames/ outputs/captions.jsonl
  omniedit_raw/ omniedit/frames/ sharegpt_raw/ sharegpt/image/
  eval_only/          robocasa/combine_all_im256.hdf5 franka_custom/data/   (planning evaluation only)
  cosmos_libero_raw/ cosmos_libero/{jsonl,stats.json}
  cosmos_robocasa_raw/ cosmos_robocasa/{jsonl,stats.json}
  robocasa_mg_raw/ robocasa_mg/{hdf5,jsonl,stats_300.json}
```

`specs/<dataset>_v1.yaml` (action/proprio mean, std, clip bounds) and `lang_bins/<dataset>_v1.yaml` (quantile bins
and phrases for the qualitative descriptions) are the normalization used by the paper. The builders read them and
abort if the rebuilt data disagree with them; `--stats-only` regenerates them from scratch.

## LaLa-World

| dataset | source | download | built |
|---|---|---|---|
| Wall, PointMaze, Push-T, Meta-World | [facebook/jepa-wms](https://huggingface.co/datasets/facebook/jepa-wms) | 6 GB (zips) | 97 GB raw, 43 GB frames |
| Rope, Granular | [DINO-WM OSF release](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28) | 16 GB (split zip) | 0.7 GB frames |
| DROID | [lerobot/droid_1.0.1](https://huggingface.co/datasets/lerobot/droid_1.0.1) (meta, data, `exterior_1_left`) | 165 GB | 149 GB frames |
| OpenVid-1M sequences | [junha1125/openvid-frame-sequences-1M](https://huggingface.co/datasets/junha1125/openvid-frame-sequences-1M) | 54 GB | 53 GB frames |
| OmniEdit | [TIGER-Lab/OmniEdit-Filtered-1.2M](https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M), every 4th shard | 745 GB | 32 GB frames |
| ShareGPT-4o-Image | [FreedomIntelligence/ShareGPT-4o-Image](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image), I2I split | 171 GB | 161 GB images |
| RoboCasa demos, Franka trajectories (evaluation only) | [facebook/jepa-wms](https://huggingface.co/datasets/facebook/jepa-wms) | 5 GB | - |

**Simulation** (`sim`): one transition = 5 simulator steps (Wall, PointMaze, Push-T, Meta-World) or one push (Rope,
Granular); frames are stored densely and every window start is a transition.

```bash
python data_preparation/download_sim.py --root data                        # needs zip/unzip for the OSF split archive
for env in wall point_maze pusht metaworld rope granular; do
  python data_preparation/build_sim_traj.py --env $env --root data --workers 24
done
```

**DROID** (`droid`): every 4th frame of the 15 fps `exterior_1_left` video, one transition per retained frame,
action = end-effector delta (xyz, euler, gripper).

```bash
python data_preparation/droid/download_droid.py --root data
python data_preparation/droid/build_droid_traj.py --root data --workers 48     # about 1 hour with 48 workers
python data_preparation/droid/validate_droid_traj.py --root data
```

**OpenVid-1M** (`openvid`): ten frames 0.5 s apart per sequence; the condition for each adjacent pair is a change
caption written by Qwen3-VL-30B-A3B-Instruct. The pre-extracted frames are on the Hub; the captions will be added
to the same repository. To regenerate them, serve the VLM behind an OpenAI-compatible endpoint:

```bash
python data_preparation/openvid/download_openvid.py --root data
export OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=none
python data_preparation/openvid/caption_sequences.py --root data --model Qwen/Qwen3-VL-30B-A3B-Instruct --workers 32   # 1,049,364 requests
python data_preparation/openvid/build_openvid_traj.py --root data --check-frames all
python data_preparation/openvid/validate_openvid_traj.py --root data
```

`openvid/raw/` rebuilds the frame sequences from the original OpenVid-1M videos (see its README).

**OmniEdit, ShareGPT-4o-Image** (`edit`): one line per edit pair; the edit instruction is the condition.

```bash
python data_preparation/edit/download_omniedit.py --root data            # --all for every shard
python data_preparation/edit/build_omniedit_pair.py --root data --workers 32
python data_preparation/edit/download_sharegpt.py --root data
python data_preparation/edit/build_sharegpt_pair.py --root data
```

**Planning-only data** (`eval`): the RoboCasa demonstrations (zero-shot planning targets) and the Franka
trajectories (DROID action score), into `data/eval_only/`.

```bash
python data_preparation/download_eval_only.py --root data
```

## LaLa-Policy

| dataset | source | download | notes |
|---|---|---|---|
| LIBERO (Cosmos Policy replays) | [nvidia/LIBERO-Cosmos-Policy](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy) | 22 GB | 1,700 demos + 2,000 rollouts, 4 suites |
| RoboCasa (Cosmos Policy replays) | [nvidia/RoboCasa-Cosmos-Policy](https://huggingface.co/datasets/nvidia/RoboCasa-Cosmos-Policy) | 152 GB | 1,199 demos + 1,291 rollouts, 24 tasks |
| RoboCasa MimicGen (300-demo protocol) | RoboCasa `mg_im` Box links (`policy/robocasa_mg_urls.tsv`) | 816 GB | 300 demos per task via the in-file `mask/300_demos`; repacked to 46 GB |

```bash
python data_preparation/policy/download_cosmos.py --root data --what libero
python data_preparation/policy/build_cosmos_libero_jsonl.py --root data
python data_preparation/policy/download_cosmos.py --root data --what robocasa
python data_preparation/policy/build_cosmos_robocasa_jsonl.py --root data
bash   data_preparation/policy/download_robocasa_mg.sh --root data 8        # 8 parallel downloads, resumable
python data_preparation/policy/verify_robocasa_mg.py --root data
python data_preparation/policy/build_robocasa_mg_jsonl.py --root data --mask 300_demos --workers 24
```

The policy JSONL records keep actions and proprio inline and point at the HDF5 files for images; `stats.json`
(`stats_300.json` for MimicGen) holds the min/max used for normalization.
