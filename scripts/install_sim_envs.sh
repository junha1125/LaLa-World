#!/usr/bin/env bash
# Simulator environments that need their own venv (they pin a different mujoco / robosuite than the planning stack).
#   bash scripts/install_sim_envs.sh robocasa   # RoboCasa v0.2 + robosuite 1.5.1 + kitchen assets (~20 GB)   -> third_party/venv-robocasa
#   bash scripts/install_sim_envs.sh libero     # LIBERO + robosuite 1.4.1 + mujoco 3.3.2                       -> third_party/venv-libero
#   bash scripts/install_sim_envs.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
unset PIP_CONSTRAINT || true
TP=${LALA_THIRD_PARTY:-$PWD/third_party}
mkdir -p "$TP"
NUMPY_PIN=$(python -c "import numpy; print('numpy==' + numpy.__version__)")
WHAT=${1:-all}
MAIN_SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
link_main_site() {  # make the main environment's packages (torch, transformers, this repo) visible inside the venv
  local sub; sub=$("$1/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  echo "import site; site.addsitedir('$MAIN_SITE')" > "$sub/lala_main_env.pth"
}

if [ "$WHAT" = robocasa ] || [ "$WHAT" = all ]; then
  # robosuite / robocasa runtime deps (installed --no-deps below to keep this env's numpy / mujoco)
  python -m pip install -c <(echo "$NUMPY_PIN") termcolor numba scipy mink pynput tqdm pyyaml imageio h5py pygame lxml opencv-python-headless Pillow
  python -m pip install --no-deps robosuite==1.5.1
  [ -d "$TP/robocasa" ] || git clone -b v0.2 https://github.com/robocasa/robocasa "$TP/robocasa"
  # robocasa v0.2 asserts mujoco == 3.2.6 and a numpy range at import; relax both (the venv below pins mujoco)
  python - "$TP/robocasa/robocasa/__init__.py" <<'PY'
import re, sys, pathlib
f = pathlib.Path(sys.argv[1]); s = f.read_text()
s = re.sub(r"assert \(\s*mujoco\.__version__.*?\n\), \"MuJoCo[^\n]*\n", "pass\n", s, flags=re.S)
s = re.sub(r"assert numpy\.__version__ in \[.*?\][^\n]*\n", "pass\n", s, flags=re.S)
f.write_text(s)
PY
  python -m pip install --no-deps -e "$TP/robocasa"
  [ -d "$TP/robocasa/robocasa/models/assets/objects/objaverse" ] || python "$TP/robocasa/robocasa/scripts/download_kitchen_assets.py" < <(printf 'y\n%.0s' {1..20})
  [ -x "$TP/venv-robocasa/bin/python" ] || python -m venv --system-site-packages "$TP/venv-robocasa"
  link_main_site "$TP/venv-robocasa"
  "$TP/venv-robocasa/bin/pip" install -c <(echo "$NUMPY_PIN") mujoco==3.2.6
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$TP/venv-robocasa/bin/python" -c "import robosuite, robocasa, mujoco; print('robocasa ok, mujoco', mujoco.__version__)"
fi

if [ "$WHAT" = libero ] || [ "$WHAT" = all ]; then
  [ -d "$TP/LIBERO" ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$TP/LIBERO"
  [ -x "$TP/venv-libero/bin/python" ] || python -m venv --system-site-packages "$TP/venv-libero"
  link_main_site "$TP/venv-libero"
  "$TP/venv-libero/bin/pip" install -c <(echo "$NUMPY_PIN") "robosuite==1.4.1" "mujoco==3.3.2" "bddl==1.0.1" easydict future thop "hydra-core>=1.2,<1.4"
  "$TP/venv-libero/bin/pip" install --no-deps -e "$TP/LIBERO"
  echo "$TP/LIBERO" > "$("$TP/venv-libero/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/libero_repo.pth"
  mkdir -p ~/.libero
  cat > ~/.libero/config.yaml <<CFG
benchmark_root: $TP/LIBERO/libero/libero
bddl_files: $TP/LIBERO/libero/libero/bddl_files
init_states: $TP/LIBERO/libero/libero/init_files
datasets: $TP/datasets
assets: $TP/LIBERO/libero/libero/assets
CFG
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$TP/venv-libero/bin/python" -c "import robosuite, mujoco, libero; print('libero ok, robosuite', robosuite.__version__, 'mujoco', mujoco.__version__)"
fi
