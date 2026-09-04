#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_DIR="${1:-outputs/dnn_5traj_$(date +%Y%m%d_%H%M%S)}"
SEEDS=(1 2 3 4 5)
mkdir -p "$RUN_DIR"

echo "Output: $RUN_DIR"

for seed in "${SEEDS[@]}"; do
    trajectory_dir="$RUN_DIR/trajectory_$seed"
    final_checkpoint="$trajectory_dir/mnist_checkpoints/epoch_100.pt"
    if [[ -s "$final_checkpoint" ]]; then
        echo "Skipping completed training trajectory $seed"
    else
        python -u dnn_trajectories_run.py \
            --seed "$seed" \
            --output-dir "$trajectory_dir"
    fi
done

python -u mlp_experiment_run.py \
    --output-dir "$RUN_DIR/main" \
    --checkpoint-root "$RUN_DIR" \
    --trajectories "${SEEDS[@]}"

python -u mlp_probe.py \
    --output-dir "$RUN_DIR/h_sweep" \
    --checkpoint-root "$RUN_DIR" \
    --trajectories "${SEEDS[@]}"

echo "Finished: $RUN_DIR"
