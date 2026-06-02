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
#   * dm_control==1.0.28 (wheelhouse) + mujoco==3.3.0 (PyPI prebuilt wheel; not
#     in the Alliance wheelhouse). Modern stack, python 3.11. The legacy py3.8
#     dm_control 0.0.x is incompatible with this interpreter, so the `reacherloca`
#     task is vendored into the repo (embodied/envs/reacherloca_task.py +
#     reacherloca.xml) and built directly — no patched dm_control / suite-domain
#     registration required.
#   * danijar libs (elements/ninjax/portal/scope/granular) are pure Python and
#     not in the wheelhouse -> PyPI --no-deps.
#   * gym_minigrid + classic gym for the MiniGrid LoCA env.
#
# NOTE: jax 0.5.0 needs Python >= 3.10, so this targets StdEnv/2023 + python/3.11
# (NOT the python/3.8 that r2dreamer used). Modules here MUST match _common.sh.
# Verify wheelhouse availability first: `avail_wheels jax dm_control mujoco`.

set -e -o pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export DREAMER_SRC="$REPO"
cd "$REPO"

# 1) Modules — must match SLURMs/_common.sh.
module --force purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load ffmpeg
module load glfw 2>/dev/null || echo "[install] glfw module not loaded (using MUJOCO_GL=egl)"

# 2) Modern mujoco (the pip `mujoco` package, installed below) renders headless
#    via EGL — no native MuJoCo 2.1.0 staging needed.
export MUJOCO_GL=egl

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

# 6) Modern dm_control + mujoco (python 3.11). The vendored reacherloca task
#    builds on this stock install (no suite-domain patch).
#    mujoco is NOT in the Alliance wheelhouse, but its PyPI release is a
#    self-contained prebuilt manylinux2014 wheel (bundles libmujoco + EGL/GLFW/
#    OSMesa) that needs no compilation, so we pull it from PyPI. dm_control comes
#    from the wheelhouse; install mujoco FIRST so dm_control's dep is satisfied.
echo "[install] Pass 3: mujoco 3.3.0 (PyPI prebuilt wheel)"
pip install --no-index "mujoco==3.3.0" 2>/dev/null \
    || pip install "mujoco==3.3.0" \
    || { echo "[install] mujoco wheel install failed. If Alliance pip rejects the"   >&2; \
         echo "          manylinux tag, download it on a login node and install the" >&2; \
         echo "          file directly, e.g.:"                                        >&2; \
         echo "            pip download --no-deps mujoco==3.3.0 -d /tmp/mj"           >&2; \
         echo "            pip install /tmp/mj/mujoco-3.3.0-*.whl"                     >&2; \
         exit 1; }

echo "[install] Pass 3b: dm_control 1.0.28 (wheelhouse; mujoco already satisfied)"
pip install --no-index "dm_control==1.0.28" "tqdm" \
    || pip install "dm_control==1.0.28" "tqdm" \
    || { echo "[install] dm_control install failed - check 'avail_wheels dm_control'" >&2; exit 1; }

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
import dm_control, mujoco
print(f"dm_control {dm_control.__version__}  mujoco {mujoco.__version__}")

# Smoke-test that dm_control + mujoco render headless via EGL.
from dm_control import suite
env = suite.load("reacher", "easy", task_kwargs={"random": 0})
env.reset()
img = env.physics.render(64, 64, camera_id=0)
print(f"dm_control reacher.easy render ok; image shape = {img.shape}")

# Smoke-test the vendored reacherloca task builds and steps.
import os
sys.path.insert(0, os.environ.get("DREAMER_SRC", "."))
from embodied.envs import reacherloca_task
renv = reacherloca_task.easy(random=0)
renv.reset()
print("vendored reacherloca task: reset ok")

if fail:
    sys.exit(1)
PY

echo "[install] done."
