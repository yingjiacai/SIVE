"""Run the end-to-end MNIST checkpoint comparison used by the paper."""

import argparse
import glob
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import resolve_sgld_config
from src.estimators import (
    apply_burn_in,
    compute_llc_naive_mean,
    compute_llc_naive_mean_specific_L,
    compute_llc_raw_variance,
    compute_linear_path_decomposition,
    compute_sive_clipped,
    compute_sive_unclipped,
)
from src.models import MlpModel
from src.provenance import write_run_manifest
from src.random_streams import make_probe_rng_streams, probe_seed_manifest
from src.sampler import get_localization_radius, run_localized_sgld


METRICS = [
    "Online",
    "Retrospective",
    "Raw_Variance",
    "SIVE_unclipped",
    "SIVE_clipped",
    "SIVE_detrended_unclipped",
    "SIVE_detrended_clipped",
    "Linear_Path_Variance",
    "Linear_Residual_Covariance",
    "Decomposition_Error",
    "Full_Gradient_Norm_Sq",
    "Gradient_Norm_Sq",
    "Slope_Proxy",
    "Scale_Multiplier_c_h",
    "Physical_h",
    "Effective_h",
    "Tether_Step",
    "RMS_Displacement",
    "Relative_Displacement",
]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_config(raw_config):
    config = resolve_sgld_config(raw_config)
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("The configuration requests CUDA, but CUDA is unavailable.")
    config["device"] = requested
    return config


def get_burned_l_values(sgld_history, config):
    values = np.asarray(sgld_history["L_bar_m"])
    if config.get("apply_burn_in", False):
        start = int(len(values) * config.get("burn_in_ratio", 0.3))
        values = values[start:]
    return values


def discover_checkpoints(checkpoint_dir, checkpoint_interval):
    checkpoints = {}
    for path in sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt"))):
        epoch = int(os.path.basename(path).replace("epoch_", "").replace(".pt", ""))
        if epoch % checkpoint_interval == 0:
            checkpoints[epoch] = path
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    return checkpoints


def run_single_sgld(config, checkpoints, seed, model, trajectory=0):
    """Run one independent SGLD trial across one training trajectory."""
    set_seed(seed)
    records = []

    for epoch in tqdm(sorted(checkpoints), desc="  SGLD checkpoints", unit="epoch", leave=False):
        checkpoint = torch.load(checkpoints[epoch], map_location=config["device"])
        theta = checkpoint["theta"].detach().to(config["device"])

        gradient = model.estimate_full_gradient(
            theta,
            batch_size=config.get("full_gradient_batch_size", 1024),
        )
        gradient_norm_sq = torch.dot(gradient, gradient).item()
        physical_h = get_localization_radius(theta, config)
        c_h = float(config.get("c_h", config.get("h")))
        slope_proxy = (config["t"] ** 2) * (physical_h ** 2) * gradient_norm_sq
        rng_streams = make_probe_rng_streams(
            seed,
            epoch,
            config["device"],
            trajectory=trajectory,
        )

        use_detrending = config.get("record_linear_detrending", True)
        history = run_localized_sgld(
            model,
            theta,
            config,
            reference_gradient=gradient if use_detrending else None,
            rng_streams=rng_streams,
        )
        records.append({
            "epoch": epoch,
            "checkpoint": checkpoint,
            "history": history,
            "gradient_norm_sq": gradient_norm_sq,
            "c_h": c_h,
            "physical_h": physical_h,
            "slope_proxy": slope_proxy,
            "tether_step": config["lr"] / (physical_h ** 2),
            "stream_seeds": probe_seed_manifest(
                seed,
                epoch,
                trajectory=trajectory,
            ),
        })

    # This remains an end-to-end retrospective baseline, not an oracle path
    # target.  It uses the last checkpoint's sampled minimum.
    final_loss_floor = np.min(get_burned_l_values(records[-1]["history"], config))

    rows = []
    histories = {}
    for record in records:
        history = record["history"]
        sive_unclipped = compute_sive_unclipped(history, config)
        sive_clipped = compute_sive_clipped(history, config)
        raw_variance = compute_llc_raw_variance(history, config)

        if "L_bar_detrended_m" in history:
            decomposition = compute_linear_path_decomposition(history, config)
            detrended_unclipped = decomposition["residual_sive_unclipped"]
            detrended_clipped = compute_sive_clipped(
                history, config, loss_key="L_bar_detrended_m"
            )
            linear_path_variance = decomposition["linear_path_variance"]
            linear_residual_covariance = decomposition[
                "linear_residual_covariance"
            ]
            decomposition_error = decomposition["decomposition_error"]
        else:
            detrended_unclipped = float("nan")
            detrended_clipped = float("nan")
            linear_path_variance = float("nan")
            linear_residual_covariance = float("nan")
            decomposition_error = float("nan")

        retained = apply_burn_in(
            history,
            config.get("burn_in_ratio", 0.3),
        ) if config.get("apply_burn_in", False) else history
        rms_displacement = float(np.mean(retained["displacement_rms_m"]))
        relative_displacement = float(np.mean(retained["relative_displacement_m"]))

        checkpoint = record["checkpoint"]
        epoch = record["epoch"]
        rows.append({
            "Epoch": epoch,
            "Train_Loss": checkpoint["train_loss"],
            "Test_Loss": checkpoint["test_loss"],
            "Online": compute_llc_naive_mean(history, config),
            "Retrospective": compute_llc_naive_mean_specific_L(
                history, config, final_loss_floor
            ),
            "Raw_Variance": raw_variance,
            "SIVE_unclipped": sive_unclipped,
            "SIVE_clipped": sive_clipped,
            "SIVE_detrended_unclipped": detrended_unclipped,
            "SIVE_detrended_clipped": detrended_clipped,
            "Linear_Path_Variance": linear_path_variance,
            "Linear_Residual_Covariance": linear_residual_covariance,
            "Decomposition_Error": decomposition_error,
            "Full_Gradient_Norm_Sq": record["gradient_norm_sq"],
            # Compatibility alias for scripts written before the full-data fix.
            "Gradient_Norm_Sq": record["gradient_norm_sq"],
            "Slope_Proxy": record["slope_proxy"],
            "Scale_Multiplier_c_h": record["c_h"],
            "Physical_h": record["physical_h"],
            "Effective_h": record["physical_h"],
            "Tether_Step": record["tether_step"],
            "RMS_Displacement": rms_displacement,
            "Relative_Displacement": relative_displacement,
            "Gradient_Stream_Seed": record["stream_seeds"]["gradient"],
            "Langevin_Stream_Seed": record["stream_seeds"]["langevin"],
            "Evaluation_Stream_Seed": record["stream_seeds"]["evaluation"],
            # Compatibility alias for older plotting notebooks.
            "Ours": sive_unclipped,
        })
        histories[epoch] = history

    return rows, histories


def save_trial(traj_out_dir, trial, rows, histories):
    pd.DataFrame(rows).to_csv(
        os.path.join(traj_out_dir, f"raw_trial_{trial}.csv"), index=False
    )
    arrays = {}
    for epoch, history in histories.items():
        for field, values in history.items():
            arrays[f"epoch_{epoch}_{field}"] = np.asarray(values)
    np.savez_compressed(
        os.path.join(traj_out_dir, f"sgld_trial_{trial}.npz"), **arrays
    )


def read_sgld_npz(npz_path):
    """Return ``epoch -> field -> array`` from a saved SGLD trial."""
    raw = np.load(npz_path)
    result = {}
    for key in raw.files:
        _, epoch_text, field = key.split("_", 2)
        result.setdefault(int(epoch_text), {})[field] = raw[key]
    return result


def summarize_trials(traj_out_dir):
    raw_files = sorted(glob.glob(os.path.join(traj_out_dir, "raw_trial_*.csv")))
    if not raw_files:
        raise FileNotFoundError(f"No raw trial files found in {traj_out_dir}")

    frames = []
    for path in raw_files:
        frame = pd.read_csv(path)
        if "SIVE_unclipped" not in frame and "Ours" in frame:
            frame["SIVE_unclipped"] = frame["Ours"]
        if "SIVE_clipped" not in frame:
            frame["SIVE_clipped"] = frame["SIVE_unclipped"].clip(lower=0)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)

    rows = []
    for epoch, group in combined.groupby("Epoch", sort=True):
        row = {
            "Epoch": int(epoch),
            "Train_Loss": group["Train_Loss"].iloc[0],
            "Test_Loss": group["Test_Loss"].iloc[0],
        }
        for metric in METRICS:
            if metric not in group:
                continue
            values = group[metric].dropna().to_numpy()
            if len(values) == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
            else:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
        for metric in ("SIVE_unclipped", "SIVE_detrended_unclipped"):
            if metric not in group:
                continue
            values = group[metric].dropna().to_numpy()
            if len(values) == 0:
                continue
            row[f"{metric}_negative_fraction"] = float(np.mean(values < 0))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
        row["Ours_mean"] = row["SIVE_unclipped_mean"]
        row["Ours_std"] = row["SIVE_unclipped_std"]
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(traj_out_dir, "experiment_results.csv"), index=False)
    return summary


def print_summary(summary):
    print(
        f"\n  {'Epoch':>6} {'TrainLoss':>10} {'TestLoss':>10} "
        f"{'Online':>12} {'Retro':>12} {'SIVE':>12} {'Detrended':>12}"
    )
    print("-" * 94)
    for _, row in summary.iterrows():
        print(
            f"  {int(row['Epoch']):>6} "
            f"{row['Train_Loss']:>10.4f} {row['Test_Loss']:>10.4f} "
            f"{row['Online_mean']:>12.4f} {row['Retrospective_mean']:>12.4f} "
            f"{row['SIVE_unclipped_mean']:>12.4f} "
            f"{row.get('SIVE_detrended_unclipped_mean', float('nan')):>12.4f}"
        )


def run_experiment(config, output_dir=None, start_trial=0):
    num_trajectories = config.get("num_trajectories", 1)
    num_trials = config.get("num_trials", 5)

    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/main_experiment_{stamp}"
        os.makedirs(output_dir, exist_ok=False)
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
        write_run_manifest(
            output_dir,
            config,
            seeds=range(num_trials),
            source_root=os.path.dirname(os.path.abspath(__file__)),
        )
    elif not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    print(f"Config: {config}")
    print(f"Output: {output_dir}")

    model = MlpModel(root=config.get("data_root", "./data"), config=config)
    for trajectory in range(num_trajectories):
        checkpoint_dir = f"outputs/trajectory_{trajectory}/mnist_checkpoints"
        checkpoints = discover_checkpoints(
            checkpoint_dir, config.get("checkpoint_interval", 1)
        )
        traj_out_dir = os.path.join(output_dir, f"trajectory_{trajectory}")
        os.makedirs(traj_out_dir, exist_ok=True)

        for trial in tqdm(
            range(start_trial, num_trials),
            desc=f"Trajectory {trajectory} trials",
            unit="trial",
        ):
            rows, histories = run_single_sgld(
                config,
                checkpoints,
                trial,
                model,
                trajectory=trajectory,
            )
            save_trial(traj_out_dir, trial, rows, histories)

        summary = summarize_trials(traj_out_dir)
        print_summary(summary)
        print(f"Saved: {os.path.join(traj_out_dir, 'experiment_results.csv')}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", default=None, help="Existing output directory")
    parser.add_argument("--start-trial", type=int, default=0)
    args = parser.parse_args()

    if args.resume:
        with open(os.path.join(args.resume, "config.json"), encoding="utf-8") as handle:
            config = prepare_config(json.load(handle))
        run_experiment(config, output_dir=args.resume, start_trial=args.start_trial)
        return

    with open("experiment_settings.json", encoding="utf-8") as handle:
        settings = json.load(handle)
    config = prepare_config(settings["4-2-1"])
    config["experiment_name"] = "4-2-1"
    run_experiment(config, start_trial=args.start_trial)


if __name__ == "__main__":
    main()
