import os
import sys
import json
import glob
import random
import argparse
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


def load_existing_trial_data(traj_out_dir, before_trial=None):
    """Load trial data from existing raw_trial_*.csv files in a trajectory directory.

    Args:
        traj_out_dir: path to trajectory output directory
        before_trial: if set, only load trials with index < before_trial
                      (i.e., skip trials that will be re-run)

    Returns:
        trials_data: {epoch: {'Online': [...], 'Ours': [...], 'Retrospective': [...]}}
        loss_data: {epoch: (train_loss, test_loss)}
    """
    trials_data = {}
    loss_data = {}

    pattern = os.path.join(traj_out_dir, "raw_trial_*.csv")
    csv_files = sorted(glob.glob(pattern))

    for csv_path in csv_files:
        basename = os.path.basename(csv_path)
        # "raw_trial_N.csv" → N
        trial_idx = int(basename.replace("raw_trial_", "").replace(".csv", ""))

        if before_trial is not None and trial_idx >= before_trial:
            continue

        df = pd.read_csv(csv_path)

        # Initialize trials_data on first file
        if not trials_data:
            for epoch in df['Epoch'].values:
                trials_data[int(epoch)] = {
                    'Online': [], 'Ours': [], 'Retrospective': []
                }

        for _, row in df.iterrows():
            epoch = int(row['Epoch'])
            if epoch not in trials_data:
                continue
            trials_data[epoch]['Online'].append(row['Online'])
            trials_data[epoch]['Ours'].append(row['Ours'])
            trials_data[epoch]['Retrospective'].append(row['Retrospective'])

        # Loss data from trial 0 (checkpoint-level values, deterministic)
        if trial_idx == 0:
            for _, row in df.iterrows():
                loss_data[int(row['Epoch'])] = (
                    row['Train_Loss'], row['Test_Loss']
                )

    return trials_data, loss_data


def resume_experiment(out_dir, start_trial):
    """Resume an interrupted experiment from a given trial index.

    Args:
        out_dir: path to an existing output directory (e.g. outputs/main_experiment_20260619_125458)
        start_trial: trial index to start/resume from (0-based)
    """
    config_path = os.path.join(out_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {out_dir}")
        sys.exit(1)

    config = json.load(open(config_path))
    # Restore device
    if 'device' in config and config['device']:
        config['device'] = torch.device(config['device'])
    else:
        config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_trajectories = config.get('num_trajectories', 1)
    num_trials = config.get('num_trials', 5)

    if start_trial >= num_trials:
        print(f"start_trial={start_trial} >= num_trials={num_trials}, nothing to do.")
        return

    print(f"Resuming experiment in: {out_dir}")
    print(f"  Config: {config}")
    print(f"  Trajectories: {num_trajectories}, Trials: {start_trial} → {num_trials - 1} "
          f"({num_trials - start_trial} to run)")

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

        # Load data from trials that ran before the interruption
        trials_data, loss_data = load_existing_trial_data(
            traj_out_dir, before_trial=start_trial
        )

        # If no data loaded yet (start_trial == 0), initialize fresh
        if not trials_data:
            for epoch in sorted(checkpoints.keys()):
                trials_data[epoch] = {
                    'Online': [], 'Ours': [], 'Retrospective': []
                }

        print(f"  Loaded {len(loss_data)} epochs of loss data, "
              f"{len(trials_data[sorted(trials_data.keys())[0]]['Online']) if trials_data else 0} "
              f"existing trials")

        for t in tqdm(range(start_trial, num_trials),
                       desc=f"  SGLD trials", unit="trial", leave=False):
            rows, sgld_histories = run_single_sgld(config, checkpoints, seed=t)
            for row in rows:
                epoch = row['Epoch']
                trials_data[epoch]['Online'].append(row['Online'])
                trials_data[epoch]['Ours'].append(row['Ours'])
                trials_data[epoch]['Retrospective'].append(row['Retrospective'])
                # Grab loss data from the first trial we encounter if missing
                if not loss_data:
                    loss_data[epoch] = (row['Train_Loss'], row['Test_Loss'])

            trial_df = pd.DataFrame(rows)
            trial_csv = os.path.join(traj_out_dir, f"raw_trial_{t}.csv")
            trial_df.to_csv(trial_csv, index=False)

            npz_data = {}
            for epoch, hist in sgld_histories.items():
                npz_data[f"epoch_{epoch}_L_bar_m"] = np.asarray(hist['L_bar_m'])
                npz_data[f"epoch_{epoch}_s2_m"] = np.asarray(hist['s2_m'])
                npz_data[f"epoch_{epoch}_L_true_m"] = np.asarray(hist['L_true_m'])
            npz_path = os.path.join(traj_out_dir, f"sgld_trial_{t}.npz")
            np.savez_compressed(npz_path, **npz_data)

        # Rebuild summary from all available trials
        summary_rows = []
        for epoch in sorted(checkpoints.keys()):
            epoch_data = trials_data[epoch]
            train_loss, test_loss = loss_data.get(
                epoch, (float('nan'), float('nan'))
            )
            n_trials = len(epoch_data['Online'])
            summary_rows.append({
                'Epoch': epoch,
                'Train_Loss': train_loss,
                'Test_Loss': test_loss,
                'Online_mean': np.mean(epoch_data['Online']),
                'Online_std': np.std(epoch_data['Online'], ddof=1) if n_trials > 1 else 0.0,
                'Retrospective_mean': np.mean(epoch_data['Retrospective']),
                'Retrospective_std': np.std(epoch_data['Retrospective'], ddof=1) if n_trials > 1 else 0.0,
                'Ours_mean': np.mean(epoch_data['Ours']),
                'Ours_std': np.std(epoch_data['Ours'], ddof=1) if n_trials > 1 else 0.0,
            })

        traj_df = pd.DataFrame(summary_rows)

        print(f"\n  {'Epoch':>6}  {'TrainLoss':>10}  {'TestLoss':>10}  "
              f"{'Online':>22} {'Retrospective':>22} {'Ours':>22}")
        print("-" * 100)
        for _, row in traj_df.iterrows():
            online_std = row['Online_std'] if row['Online_std'] is not None else 0.0
            ret_std = row['Retrospective_std'] if row['Retrospective_std'] is not None else 0.0
            ours_std = row['Ours_std'] if row['Ours_std'] is not None else 0.0
            print(f"  {int(row['Epoch']):>6}  "
                  f"{row['Train_Loss']:>10.4f}  "
                  f"{row['Test_Loss']:>10.4f}  "
                  f"{row['Online_mean']:>12.4f} +/- {online_std:<5.4f}  "
                  f"{row['Retrospective_mean']:>12.4f} +/- {ret_std:<5.4f}"
                  f"{row['Ours_mean']:>12.4f} +/- {ours_std:<5.4f}  ")

        traj_csv = os.path.join(traj_out_dir, "experiment_results.csv")
        traj_df.to_csv(traj_csv, index=False)
        if start_trial > 0:
            print(f"\n  Saved: {traj_csv}  "
                  f"({start_trial} existing + {num_trials - start_trial} new = "
                  f"{num_trials} trials total)")
        else:
            print(f"\n  Saved: {traj_csv}  ({num_trials} trials total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLC experiment with optional resume support"
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume from an existing output directory '
             '(e.g. outputs/main_experiment_20260619_125458)'
    )
    parser.add_argument(
        '--start-trial', type=int, default=0,
        help='Trial index to start from (0-based, default: 0). '
             'In resume mode, loads existing trials before this index. '
             'In normal mode, overrides the default start of 0.'
    )
    args = parser.parse_args()

    if args.resume:
        # ── Resume mode ──────────────────────────────────────────
        resume_experiment(args.resume, args.start_trial)

    else:
        # ── Normal (fresh) mode ──────────────────────────────────
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
