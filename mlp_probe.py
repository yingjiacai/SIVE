import os
import json
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models import MlpModel
from src.sampler import run_localized_sgld
from src.estimators import compute_llc_naive_mean, compute_llc_debiased_variance

CHECKPOINT_DIR = "outputs/trajectory_0/mnist_checkpoints"
EPOCH_CHECK_LIST = [1, 29, 100]
additional_setting_set = {
    'h': [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0],
}


def get_config():
    with open("experiment_settings.json") as f:
        settings = json.load(f)
    config = settings["4-2-1"]
    config['t'] = config['n'] * config['beta']
    config['lr'] = config['base_lr'] / config['t']
    if 'device' not in config:
        config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    return config


def print_segment_llc(sgld_history, config, epoch, n_segments=10):
    if n_segments <= 0:
        raise ValueError("n_segments must be positive.")

    L_bar = np.asarray(sgld_history['L_bar_m'])
    s2 = np.asarray(sgld_history['s2_m'])
    L_true = np.asarray(sgld_history['L_true_m'])
    seg_len = len(L_bar) // n_segments
    if seg_len == 0:
        raise ValueError("SGLD history is shorter than the requested number of segments.")

    segment_config = config.copy()
    segment_config['apply_burn_in'] = False
    naive_llcs = []
    ours_llcs = []
    means = []
    variances = []
    for i in range(n_segments):
        seg = {
            'L_bar_m': L_bar[i * seg_len:(i + 1) * seg_len],
            's2_m': s2[i * seg_len:(i + 1) * seg_len],
            'L_true_m': L_true[i * seg_len:(i + 1) * seg_len],
        }
        naive_llcs.append(compute_llc_naive_mean(seg, segment_config))
        ours_llcs.append(compute_llc_debiased_variance(seg, segment_config))
        means.append(np.mean(seg['L_bar_m']))
        variances.append(np.var(seg['L_bar_m']))

    print(f"\n=== Epoch {epoch:>3} per-segment LLC ===")
    for row in range(2):
        idx = row * 5
        cells = [f"  mean={means[i]:.4f}, var={variances[i]:.6f}  " for i in range(idx, idx + 5)]
        label = f"  seg{idx}-{idx+4}  |"
        print(label + "".join(cells))
    cells_naive = "".join(f"  {v:>8.4f}" for v in naive_llcs)
    print("  naive |" + cells_naive)
    cells_ours = "".join(f"  {v:>8.4f}" for v in ours_llcs)
    print("  ours  |" + cells_ours)


def build_run_configs(base_config):
    """Zip param lists sequentially (not Cartesian product)."""
    keys = list(additional_setting_set.keys())
    values = list(additional_setting_set.values())
    lengths = {len(v) for v in values}
    if len(lengths) > 1:
        raise ValueError(f"All param lists must have same length for sequential sweep, got {lengths}")
    n = lengths.pop()
    configs = []
    for i in range(n):
        cfg = base_config.copy()
        for k, v_list in zip(keys, values):
            cfg[k] = v_list[i]
        configs.append((f"run_{i}", cfg))
    return configs


def probe_checkpoints(model, checkpoints, config, out_dir):
    hist_dir = os.path.join(out_dir, "sgld_histories")
    os.makedirs(hist_dir, exist_ok=True)
    summary_rows = []

    for epoch in sorted(checkpoints.keys()):
        ckpt = torch.load(checkpoints[epoch], map_location=config['device'])
        theta = ckpt['theta']
        train_loss = ckpt['train_loss']
        test_loss = ckpt['test_loss']

        sgld_history = run_localized_sgld(model, theta, config)

        naive = compute_llc_naive_mean(sgld_history, config)
        ours = compute_llc_debiased_variance(sgld_history, config)

        print_segment_llc(sgld_history, config, epoch)

        hist_df = pd.DataFrame({
            'step': range(len(sgld_history['L_bar_m'])),
            'L_bar_m': sgld_history['L_bar_m'],
            's2_m': sgld_history['s2_m'],
            'L_true_m': sgld_history['L_true_m'],
        })
        hist_df.to_csv(os.path.join(hist_dir, f"epoch_{epoch:03d}.csv"), index=False)

        summary_rows.append({
            'Epoch': epoch,
            'Train_Loss': train_loss,
            'Test_Loss': test_loss,
            'Naive': naive,
            'Ours': ours,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "probe_summary.csv"), index=False)
    return summary_df


def main():
    base_config = get_config()

    checkpoints = {}
    for epoch in EPOCH_CHECK_LIST:
        path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoints[epoch] = path

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {CHECKPOINT_DIR}")

    run_configs = build_run_configs(base_config)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = f"outputs/probe_sweep_{ts}"
    os.makedirs(sweep_dir, exist_ok=True)

    model = MlpModel(root='./data', config=base_config)

    all_summaries = []
    for run_name, config in run_configs:
        desc = ", ".join(f"{k}={config[k]}" for k in additional_setting_set)
        print(f"\n{'='*60}\n{run_name}: {desc}\n{'='*60}")

        run_dir = os.path.join(sweep_dir, run_name)
        summary_df = probe_checkpoints(model, checkpoints, config, run_dir)
        summary_df['run'] = run_name
        all_summaries.append(summary_df)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(os.path.join(sweep_dir, "sweep_summary.csv"), index=False)

    print(f"\nDone. {len(run_configs)} runs x {len(checkpoints)} epochs.")
    print(f"Output: {sweep_dir}")


if __name__ == "__main__":
    main()
