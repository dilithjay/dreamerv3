# Shared environment setup for JAX DreamerV3 SLURM jobs. Source this from the
# generated sbatch scripts (submit.sh) after the SBATCH headers.
#
# Mirrors r2dreamer/SLURMs/_common.sh but for the JAX stack:
#   * StdEnv/2023 + python/3.11 + cuda/12.2 (jax 0.5.0 needs Python >= 3.10 and
#     pairs with the wheelhouse jax-cuda12 plugin built against cuda 12.x).
#   * Modern dm_control 1.0.28 + mujoco 3.3.0 (pip package); headless rendering
#     via EGL (MUJOCO_GL=egl). No native MuJoCo 2.1.0 / mujoco210 staging.
#
# Modules MUST match SLURMs/install_venv.sh.

set -e -o pipefail

# Gentle stagger so all array tasks don't hammer Lustre at once.
sleep $(( (SLURM_ARRAY_TASK_ID % 10) * 3 ))

module --force purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load ffmpeg
module load glfw 2>/dev/null || true  # headless MuJoCo rendering (optional)

# Source tree + venv. R2DREAMER-style layout: this file lives in <SRC>/SLURMs/.
DREAMER_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DREAMER_SRC
source "$DREAMER_SRC/.venv/bin/activate"

# Headless GPU rendering via EGL for the modern mujoco pip package.
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

# Threading caps — keep BLAS/OpenMP from oversubscribing the allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export SLURM_CPU_BIND=cores

# JAX: let it see the GPU; don't preallocate the whole device so eval/env render
# share the GPU. (Mirror upstream behavior; tune if needed.)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export WANDB_MODE=offline
export PYTHONPATH="$DREAMER_SRC:${PYTHONPATH:-}"

: "${SLURM_TMPDIR:=/tmp}"
SEED="${SLURM_ARRAY_TASK_ID}"
export SEED
