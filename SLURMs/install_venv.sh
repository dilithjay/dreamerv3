#!/bin/bash
# One-shot venv setup for the JAX DreamerV3 (+ LoFo/LoCA) on Alliance Canada.
#
# Run this ONCE on a login node:
#   cd <repo>/dreamerv3
#   bash SLURMs/install_venv.sh
#
# Strategy (mirrors r2dreamer/SLURMs/install_venv.sh, but for the JAX stack):
#   * jax==0.5.0 from the Alliance wheelhouse (--no-index). The wheelhouse
#     supplies the matching jaxlib + jax-cuda12 plugin/pjrt so CUDA/cuDNN come
#     from the wheel under the loaded cuda/12.2 module — no nvidia-* pip wheels.
#     0.5.0 is the version this dreamerv3 is tested against (Dockerfile).
#   * dm-control==0.0.403778684 (pre-1.0) from PyPI --no-deps. This loads MuJoCo
#     2.1.0 via ctypes from $MUJOCO_PATH and is the version that registers the
#     patched `reacherloca` domain (matches LoCA_v3 / Dreamer_v1_Ali).
#   * danijar libs (elements/ninjax/portal/scope/granular) are pure Python and
#     not in the wheelhouse -> PyPI --no-deps.
#   * gym_minigrid + classic gym for the MiniGrid LoCA env.
#
# NOTE: jax 0.5.0 needs Python >= 3.10, so this targets StdEnv/2023 + python/3.11
# (NOT the python/3.8 that r2dreamer used). Modules here MUST match _common.sh.

set -e -o pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# 1) Modules — must match SLURMs/_common.sh.
module --force purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load ffmpeg
module load glfw 2>/dev/null || echo "[install] glfw module not loaded (using MUJOCO_GL=egl)"

# 2) Stage MuJoCo 2.1.0 native libraries at ~/.mujoco/mujoco210/ (DeepMind
#    release; renamed to the legacy directory name expected by old dm_control).
MUJOCO_DIR="$HOME/.mujoco/mujoco210"
if [[ ! -d "$MUJOCO_DIR/bin" ]]; then
    echo "[install] staging MuJoCo 2.1.0 at $MUJOCO_DIR"
    mkdir -p "$HOME/.mujoco"
    tmpdir="$(mktemp -d)"
    (cd "$tmpdir" \
        && curl -L -O https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco-2.1.0-linux-x86_64.tar.gz \
        && tar xzf mujoco-2.1.0-linux-x86_64.tar.gz \
        && mv mujoco-2.1.0 "$MUJOCO_DIR")
    rm -rf "$tmpdir"
fi
[[ -d "$MUJOCO_DIR/bin" ]] || { echo "MuJoCo 2.1.0 staging failed; check $MUJOCO_DIR" >&2; exit 1; }
export MUJOCO_PATH="$MUJOCO_DIR"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"

# 3) Build the venv against Alliance's Python.
if [[ ! -d .venv ]]; then
    virtualenv --no-download .venv
fi
source .venv/bin/activate
pip install --no-index --upgrade pip

# 4) Core JAX stack from the Alliance wheelhouse. `jax==0.5.0` resolves the
#    matching jaxlib + jax-cuda12 plugin/pjrt from the wheelhouse.
echo "[install] Pass 1: Alliance wheelhouse (jax + numerics)"
pip install --no-index \
    "jax==0.5.0" \
    "optax" \
    "chex" \
    "jaxtyping" \
    "numpy<2"

# 5) danijar libraries — pure Python, PyPI --no-deps (not in wheelhouse).
echo "[install] Pass 2: danijar libraries from PyPI"
for pkg in "elements>=3.19.1" "ninjax>=3.5.1" "portal>=3.5.0" "scope>=0.4.4" "granular>=0.20.3" "einops"; do
    pip install --no-index "$pkg" 2>/dev/null \
        || pip install --index-url https://pypi.org/simple --no-deps "$pkg" \
        || echo "[install]   warning: $pkg not installed"
done

# 6) Pre-1.0 dm_control via PyPI (pure Python; registers the patched reacherloca).
echo "[install] Pass 3: dm-control 0.0.x from PyPI (MuJoCo 2.1.0 via ctypes)"
pip install --index-url https://pypi.org/simple --no-deps "dm-control==0.0.403778684"
for dep in "labmaze" "lxml" "pyopengl" "glfw" "tqdm"; do
    pip install --no-index "$dep" 2>/dev/null \
        || pip install --index-url https://pypi.org/simple --no-deps "$dep" \
        || echo "[install]   warning: $dep not installed"
done

# 7) gym_minigrid 1.2.2 — pure Python, needed for the MiniGrid LoCA env. It
#    depends on classic `gym` (not gymnasium).
echo "[install] Pass 4: gym_minigrid for the MiniGrid LoCA env"
pip install --no-index "gym-minigrid==1.2.2" 2>/dev/null \
    || pip install --index-url https://pypi.org/simple --no-deps "gym-minigrid==1.2.2"
pip install --no-index "gym==0.26.2" 2>/dev/null \
    || pip install --index-url https://pypi.org/simple --no-deps "gym==0.26.2"

# 8) Verify.
echo "[install] verifying imports..."
python - <<'PY'
import importlib, sys
mods = ["jax", "jaxlib", "optax", "chex", "numpy",
        "elements", "ninjax", "portal", "dm_control", "gym_minigrid"]
fail = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ok  {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
        fail.append(m)

import jax
print(f"jax {jax.__version__}  devices = {jax.devices()}")
import numpy
print(f"numpy {numpy.__version__}")
import dm_control
print(f"dm_control {dm_control.__version__}")

# Smoke-test the patched reacherloca task loads (fails here if MUJOCO_PATH /
# LD_LIBRARY_PATH are wrong, or if reacherloca isn't registered).
from dm_control import suite
env = suite.load("reacherloca", "easy", task_kwargs={"random": 0})
ts = env.reset()
print(f"dm_control reacherloca.easy reset ok; obs keys = {list(ts.observation.keys())}")

if fail:
    sys.exit(1)
PY

echo "[install] done."
