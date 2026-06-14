import os
import json
import random
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.models import MlpModel
from src.sampler import run_localized_sgld
from src.estimators import (
    compute_llc_naive_mean,
    compute_llc_naive_mean_specific_L,
    compute_llc_debiased_variance
)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_burned_L_values(sgld_history, config):
    L_values = np.asarray(sgld_history['L_bar_m'])
    if config.get('apply_burn_in', False):
        burn_in_ratio = config.get('burn_in_ratio', 0.3)
        start = int(len(L_values) * burn_in_ratio)
        L_values = L_values[start:]
    return L_values


def run_single_sgld(config, checkpoints, seed):
    """Run one SGLD trial across all checkpoints of a trajectory.

    seed controls randomness for this SGLD run.
    Returns per-epoch result rows and the raw SGLD histories.
    """
    set_seed(seed)
    model = MlpModel(root='./data', config=config)

    sgld_histories = []
    epoch_list = []
    train_loss_list = []
    test_loss_list = []

    for epoch in tqdm(sorted(checkpoints.keys()), desc="  SGLD per-epoch", unit="epoch", leave=False):
        ckpt = torch.load(checkpoints[epoch], map_location=config.get('device', 'cpu'))
        theta = ckpt['theta']
        sgld_history = run_localized_sgld(model, theta, config)

        sgld_histories.append(sgld_history)
        epoch_list.append(epoch)
        train_loss_list.append(ckpt['train_loss'])
        test_loss_list.append(ckpt['test_loss'])

    # Retrospective baseline: empirical minimum from the final checkpoint
    final_L_values = get_burned_L_values(sgld_histories[-1], config)
    final_L = np.min(final_L_values)

    rows = []
    for i, sgld_history in enumerate(sgld_histories):
        online = compute_llc_naive_mean(sgld_history, config)
        retrospective = compute_llc_naive_mean_specific_L(sgld_history, config, final_L)
        ours = compute_llc_debiased_variance(sgld_history, config)

        rows.append({
            'Epoch': epoch_list[i],
            'Train_Loss': train_loss_list[i],
            'Test_Loss': test_loss_list[i],
            'Online': online,
            'Ours': ours,
            'Retrospective': retrospective,
        })

    return rows, dict(zip(epoch_list, sgld_histories))


def read_sgld_npz(npz_path):
    """Read a sgld_trial_*.npz file and return epoch -> {field: array} dict.

    Returns:
        {1: {'L_bar_m': ndarray, 's2_m': ndarray, 'L_true_m': ndarray}, ...}

    Usage:
        data = read_sgld_npz("outputs/.../trajectory_0/sgld_trial_0.npz")
        for epoch, arrays in data.items():
            plt.plot(arrays['L_bar_m'], label=f"epoch {epoch}")
    """
    raw = np.load(npz_path)
    result = {}
    for key in raw.files:
        # key format: "epoch_N_field"  e.g. "epoch_1_L_bar_m"
        prefix, epoch_str, field = key.split("_", 2)
        epoch = int(epoch_str)
        if epoch not in result:
            result[epoch] = {}
        result[epoch][field] = raw[key]
    return result


def run_experiment(config, num_trajectories, num_trials):
    """Run the full experiment across trajectories and independent SGLD trials.

    For each trajectory, runs num_trials independent SGLD chains, collects
    per-epoch mean +/- std for each estimator, saves per-trial raw data and
    per-trajectory summary CSVs.
    """
    print(f"Config: {config}")
    print(f"Trajectories: {num_trajectories}, Trials per trajectory: {num_trials}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"outputs/main_experiment_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    config_to_save = {k: v for k, v in config.items() if not isinstance(v, (torch.device,))}
    config_to_save['device'] = str(config.get('device', ''))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_to_save, f, indent=2)
    print(f"Output: {out_dir}\n")

    for traj in range(num_trajectories):
        print()
        print(f"{'='*70}")
        print(f"Trajectory {traj + 1}/{num_trajectories}")
        print(f"{'='*70}")

        traj_input_dir = f"outputs/trajectory_{traj}"
        traj_out_dir = os.path.join(out_dir, f"trajectory_{traj}")
        os.makedirs(traj_out_dir, exist_ok=True)
        checkpoint_dir = os.path.join(traj_input_dir, "mnist_checkpoints")

        checkpoints = {}
        for f in sorted(os.listdir(checkpoint_dir)):
            if f.startswith("epoch_") and f.endswith(".pt"):
                epoch = int(f.replace("epoch_", "").replace(".pt", ""))
                if epoch % config['checkpoint_interval'] == 0:
                    checkpoints[epoch] = os.path.join(checkpoint_dir, f)

        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

        trials_data = {epoch: {'Online': [], 'Ours': [], 'Retrospective': []}
                       for epoch in sorted(checkpoints.keys())}

        # Train/test loss is the same across trials; grab from the first one
        loss_data = {}

        for t in tqdm(range(num_trials), desc=f"  SGLD trials", unit="trial", leave=False):
            rows, sgld_histories = run_single_sgld(config, checkpoints, seed=t)
            for row in rows:
                epoch = row['Epoch']
                trials_data[epoch]['Online'].append(row['Online'])
                trials_data[epoch]['Ours'].append(row['Ours'])
                trials_data[epoch]['Retrospective'].append(row['Retrospective'])
                if t == 0:
                    loss_data[epoch] = (row['Train_Loss'], row['Test_Loss'])

            trial_df = pd.DataFrame(rows)
            trial_csv = os.path.join(traj_out_dir, f"raw_trial_{t}.csv")
            trial_df.to_csv(trial_csv, index=False)

            npz_data = {}
            for epoch, hist in sgld_histories.items():
                npz_data[f"epoch_{epoch}_L_bar_m"] = np.asarray(hist['L_bar_m'])
                npz_data[f"epoch_{epoch}_s2_m"] = np.asarray(hist['s2_m'])
                npz_data[f"epoch_{epoch}_L_true_m"] = np.asarray(hist['L_true_m'])
            np.savez_compressed(os.path.join(traj_out_dir, f"sgld_trial_{t}.npz"), **npz_data)

        summary_rows = []
        for epoch in sorted(checkpoints.keys()):
            epoch_data = trials_data[epoch]
            train_loss, test_loss = loss_data[epoch]
            summary_rows.append({
                'Epoch': epoch,
                'Train_Loss': train_loss,
                'Test_Loss': test_loss,
                'Online_mean': np.mean(epoch_data['Online']),
                'Online_std': np.std(epoch_data['Online'], ddof=1),
                'Retrospective_mean': np.mean(epoch_data['Retrospective']),
                'Retrospective_std': np.std(epoch_data['Retrospective'], ddof=1),
                'Ours_mean': np.mean(epoch_data['Ours']),
                'Ours_std': np.std(epoch_data['Ours'], ddof=1),
            })

        traj_df = pd.DataFrame(summary_rows)

        print(f"\n  {'Epoch':>6}  {'TrainLoss':>10}  {'TestLoss':>10}  {'Online':>22} {'Retrospective':>22} {'Ours':>22} ")
        print("-" * 100)
        for _, row in traj_df.iterrows():
            print(f"  {int(row['Epoch']):>6}  "
                  f"{row['Train_Loss']:>10.4f}  "
                  f"{row['Test_Loss']:>10.4f}  "
                  f"{row['Online_mean']:>12.4f} +/- {row['Online_std']:<5.4f}  "
                  f"{row['Retrospective_mean']:>12.4f} +/- {row['Retrospective_std']:<5.4f}"
                  f"{row['Ours_mean']:>12.4f} +/- {row['Ours_std']:<5.4f}  ")

        traj_csv = os.path.join(traj_out_dir, "experiment_results.csv")
        traj_df.to_csv(traj_csv, index=False)
        print(f"\n  Saved: {traj_csv}")


if __name__ == "__main__":
    experiments_list = ["4-2-1"]
    for experiment_name in experiments_list:

        experiment_settings = json.load(open("experiment_settings.json"))

        print()
        print(f"Running {experiment_name}...")

        config = experiment_settings[experiment_name]

        config['t'] = config['n'] * config['beta']
        config['lr'] = config['base_lr'] / config['t']
        config['experiment_name'] = experiment_name
        if 'device' not in config:
            config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

        num_trajectories = config.get('num_trajectories', 1)
        num_trials = config.get('num_trials', 5)
        run_experiment(config, num_trajectories, num_trials)
