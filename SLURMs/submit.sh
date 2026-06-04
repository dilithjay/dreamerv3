#!/bin/bash
# Single submitter for the JAX DreamerV3 (+ LoFo/LoCA), mirroring
# r2dreamer/SLURMs/submit.sh. Run as: bash submit.sh
#
# For each entry in CONFIGS, reads the small YAML in configs/, generates a
# temporary sbatch script, and submits it. Each generated script sources
# _common.sh (env setup) and runs dreamerv3/main.py with the YAML's flags.
#
# Unlike r2dreamer (Hydra key=value), the JAX entry point uses elements.Flags:
#   named configs via `--configs defaults <name>` and overrides via
#   `--replay.variant lofo_v1 --seed N --logdir DIR`.

set -eo pipefail

# =================================================================
# Pick configs (uncomment one or more).
# 4 envs x 3 buffers = 12 combinations.
# =================================================================
CONFIGS=(
    # --- MiniGrid (vanilla, no LoCA) ---
    configs/fifo_minigrid.yaml
    configs/v1_minigrid.yaml
    configs/v2_minigrid.yaml

    # --- MiniGrid LoCA (2-phase) ---
    # configs/fifo_minigrid_loca.yaml
    # configs/v1_minigrid_loca.yaml
    # configs/v2_minigrid_loca.yaml

    # --- Reacher (vanilla, no LoCA) ---
    # configs/fifo_reacher.yaml
    # configs/v1_reacher.yaml
    # configs/v2_reacher.yaml

    # --- Reacher LoCA (2-phase) ---
    # configs/fifo_reacher_loca.yaml
    # configs/v1_reacher_loca.yaml
    # configs/v2_reacher_loca.yaml
)

# Optional overrides applied to EVERY config in this batch (extra flag tokens).
OVERRIDE_ARRAY="0"     # e.g. "0-4"
OVERRIDE_TIME=""      # e.g. "1-23:59:59"
OVERRIDE_ARGS=()
# OVERRIDE_ARGS+=(--replay.size 300000)

# =================================================================
# Machinery below.
# =================================================================
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
RESULTS_BASE="$HOME/projects/def-rsdjjana/dilith/dreamerv3/results"
COMMON="$SCRIPT_DIR/_common.sh"
VENV=".venv"

pick_python() {
    if [ -x "$VENV/bin/python3" ]; then echo "$VENV/bin/python3";
    elif command -v python3 &>/dev/null; then echo python3;
    else echo python; fi
}

# Parse a config YAML into bash variable assignments printed to stdout.
parse_yaml() {
    "$(pick_python)" - "$1" <<'PY'
import sys, shlex
try:
    import yaml
except ImportError:
    sys.stderr.write("PyYAML missing - install it in the venv\n")
    sys.exit(1)
c = yaml.safe_load(open(sys.argv[1]))
print(f"YAML_EXP_NAME={shlex.quote(str(c['exp_name']))}")
print(f"YAML_ARRAY={shlex.quote(str(c['array']))}")
print(f"YAML_TIME={shlex.quote(str(c['time']))}")
print(f"YAML_PROFILE={shlex.quote(str(c.get('profile', 'mujoco')))}")
print(f"YAML_CONFIGS={shlex.quote(str(c['configs']))}")
# Overrides dict -> "--key value" flag tokens.
pieces = []
for k, v in (c.get('overrides') or {}).items():
    pieces += [f"--{k}", str(v)]
print("YAML_OVERRIDES=(" + " ".join(shlex.quote(p) for p in pieces) + ")")
PY
}

submit_one() {
    local config="$1"
    local config_path="$SCRIPT_DIR/$config"
    [ -f "$config_path" ] || { echo "[submit] config not found: $config_path" >&2; return 1; }

    local YAML_EXP_NAME YAML_ARRAY YAML_TIME YAML_PROFILE YAML_CONFIGS
    local -a YAML_OVERRIDES
    eval "$(parse_yaml "$config_path")"

    local exp_name="$YAML_EXP_NAME"
    local array="${OVERRIDE_ARRAY:-$YAML_ARRAY}"
    local time="${OVERRIDE_TIME:-$YAML_TIME}"
    local out_dir="$RESULTS_BASE/$exp_name"
    mkdir -p "$out_dir"

    local overrides_str
    overrides_str="$(printf '%q ' "${YAML_OVERRIDES[@]}" "${OVERRIDE_ARGS[@]}")"

    local tmp_script="$SCRIPT_DIR/.submit_${exp_name}_$(date +%s%N).sh"
    cat > "$tmp_script" <<EOF
#!/bin/bash
#SBATCH --job-name=$exp_name
#SBATCH --time=$time
#SBATCH --array=$array
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --acctg-freq=task=1
#SBATCH --output=$out_dir/%A-%a.out
#SBATCH --error=$out_dir/%A-%a.err

export DREAMER_PROFILE=$YAML_PROFILE
source "$COMMON"

FINAL_DIR="$out_dir/\$SEED"
mkdir -p "\$FINAL_DIR"
cd "\$FINAL_DIR"

python -u "dreamerv3/main.py" \\
    --configs $YAML_CONFIGS \\
    $overrides_str \\
    --seed "\$SEED" \\
    --logdir "\$FINAL_DIR"
EOF

    echo "[submit] $config  ->  $exp_name  (array=$array time=$time)"
    sbatch "$tmp_script"
    sleep 1
    rm -f "$tmp_script"
}

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "No configs selected - uncomment one or more lines in CONFIGS." >&2
    exit 1
fi

rc=0
for config in "${CONFIGS[@]}"; do
    submit_one "$config" || rc=$?
done
exit $rc
