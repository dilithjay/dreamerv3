# Shared environment setup for JAX DreamerV3 SLURM jobs. Source this from the
# generated sbatch scripts (submit.sh) after the SBATCH headers.
#
# Mirrors r2dreamer/SLURMs/_common.sh but for the JAX stack:
#   * StdEnv/2023 + python/3.11 + cuda/12.2 (jax 0.5.0 needs Python >= 3.10 and
#     pairs with the wheelhouse jax-cuda12 plugin built against cuda 12.x).
#   * Modern dm_control 1.0.28 + mujoco 3.3.0 (pip package); headless rendering
#     via OSMesa (MUJOCO_GL=osmesa, CPU software rasterizer) so MuJoCo never
#     competes with JAX for the GPU on MIG slices. No native MuJoCo 2.1.0 staging.
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
# OSMesa (libOSMesa.so) for headless CPU rendering (MUJOCO_GL=osmesa below) is
# already in the system linker cache via the Compute Canada gentoo prefix
# (/cvmfs/.../gentoo/2020/usr/lib64), so no module load is needed — and there is
# no Mesa graphics module on Narval anyway. Verify: ldconfig -p | grep -i osmesa
module load glfw 2>/dev/null || true  # only used for windowed GL; unused here

source ".venv/bin/activate"

# Headless CPU rendering via OSMesa. EGL renders on the GPU and competes with
# JAX for the device, which is unreliable on MIG slices (manifests as
# CUDA_ERROR_ILLEGAL_ADDRESS GPU faults); OSMesa is a pure software rasterizer
# that never touches the GPU. Costs some CPU per frame (fine for 64x64 reacher
# frames). Only affects MuJoCo envs (reacher); inert for MiniGrid, which already
# renders on the CPU.
export MUJOCO_GL=osmesa

# Kept from the EGL setup: /usr/lib/nvidia also provides the NVIDIA driver libs
# (e.g. libcuda.so) that JAX's CUDA backend needs, so leave it on the path even
# though OSMesa itself doesn't use it.
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
# cuda is listed first so it stays the default device for the agent; cpu must
# also be initialized so the LoFo distance model can run off the accelerator
# (see dreamerv3/state_distance.py _cpu_device). Without cpu here, _cpu_device()
# falls back to the GPU and every per-insert repr call contends with training.
export JAX_PLATFORMS=cuda,cpu

# NOTE: this export is effectively a no-op. embodied/jax/internal.py sets
# XLA_PYTHON_CLIENT_PREALLOCATE from the `jax.prealloc` config value at startup,
# overwriting whatever we set here. The defaults config has jax.prealloc: True,
# so prealloc is actually ON. To change it, pass `--jax.prealloc False` (or edit
# the config), not this line. Kept only as documentation of the knob.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export WANDB_MODE=offline

: "${SLURM_TMPDIR:=/tmp}"
SEED="${SLURM_ARRAY_TASK_ID}"
export SEED
