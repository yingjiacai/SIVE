"""Run the paired multi-scale audit on selected MNIST checkpoints."""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from mlp_experiment_run import (
    METRICS,
    discover_checkpoints,
    prepare_config,
    run_single_sgld,
    save_trial,
)
from src.models import MlpModel
from src.provenance import write_run_manifest


def format_scale(value):
    return str(value).replace("-", "m").replace(".", "p")


def summarize_sweep(rows):
    frame = pd.DataFrame(rows)
    summary_rows = []
    for (c_h, epoch), group in frame.groupby(["c_h", "Epoch"], sort=True):
        row = {
            "c_h": c_h,
            "Epoch": int(epoch),
            "Train_Loss": group["Train_Loss"].iloc[0],
            "Test_Loss": group["Test_Loss"].iloc[0],
        }
        for metric in METRICS:
            values = group[metric].dropna().to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        for metric in ("SIVE_unclipped", "SIVE_detrended_unclipped"):
            values = group[metric].dropna().to_numpy()
            if len(values) == 0:
                continue
            row[f"{metric}_negative_fraction"] = float(np.mean(values < 0))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def paired_contrasts(rows, checkpoint_epochs):
    """Compute prespecified early-middle and late-middle paired contrasts."""
    early, middle, late = sorted(checkpoint_epochs)
    frame = pd.DataFrame(rows)
    contrast_rows = []
    for (c_h, trial), group in frame.groupby(["c_h", "Trial"], sort=True):
        by_epoch = group.set_index("Epoch")
        row = {"c_h": c_h, "Trial": int(trial)}
        for metric in ("SIVE_unclipped", "SIVE_detrended_unclipped"):
            row[f"{metric}_D_drop"] = (
                by_epoch.loc[early, metric] - by_epoch.loc[middle, metric]
            )
            row[f"{metric}_D_rebound"] = (
                by_epoch.loc[late, metric] - by_epoch.loc[middle, metric]
            )
        contrast_rows.append(row)
    raw = pd.DataFrame(contrast_rows)

    summary_rows = []
    value_columns = [
        column for column in raw.columns if column not in {"c_h", "Trial"}
    ]
    for c_h, group in raw.groupby("c_h", sort=True):
        row = {"c_h": c_h}
        for column in value_columns:
            values = group[column].dropna().to_numpy()
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        summary_rows.append(row)
    return raw, pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Output directory for the sweep")
    parser.add_argument("--checkpoint-root", default="outputs")
    parser.add_argument("--trajectories", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    with open("experiment_settings.json", encoding="utf-8") as handle:
        settings = json.load(handle)

    sweep = settings["dnn_h_sweep"]
    base_name = sweep["base_experiment"]
    base_config = prepare_config(settings[base_name])
    c_h_values = sweep.get("c_h_values", sweep.get("h_values"))
    if c_h_values is None:
        raise KeyError("dnn_h_sweep must define 'c_h_values'.")
    epochs = set(sweep["checkpoint_epochs"])
    if len(epochs) != 3:
        raise ValueError("The paired drop/rebound audit requires exactly three checkpoints.")
    num_trials = int(sweep.get("num_trials", base_config.get("num_trials", 5)))
    probe_seeds = sweep.get(
        "probe_seeds",
        base_config.get("probe_seeds", list(range(num_trials))),
    )

    if args.output_dir:
        sweep_dir = args.output_dir
        os.makedirs(sweep_dir, exist_ok=True)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = f"outputs/dnn_h_sweep_{stamp}"
        os.makedirs(sweep_dir, exist_ok=False)
    manifest_path = os.path.join(sweep_dir, "run_manifest.json")
    if not os.path.exists(manifest_path):
        write_run_manifest(
            sweep_dir,
            {
                "base_config": base_config,
                "sweep": sweep,
                "training_trajectories": args.trajectories,
            },
            seeds=probe_seeds,
            source_root=".",
        )

    model = MlpModel(root=base_config.get("data_root", "./data"), config=base_config)
    for trajectory in args.trajectories:
        all_checkpoints = discover_checkpoints(
            os.path.join(
                args.checkpoint_root,
                f"trajectory_{trajectory}",
                "mnist_checkpoints",
            ),
            base_config.get("checkpoint_interval", 1),
        )
        checkpoints = {
            epoch: path for epoch, path in all_checkpoints.items() if epoch in epochs
        }
        missing = epochs.difference(checkpoints)
        if missing:
            raise FileNotFoundError(f"Missing requested checkpoints: {sorted(missing)}")

        trajectory_dir = os.path.join(sweep_dir, f"trajectory_{trajectory}")
        os.makedirs(trajectory_dir, exist_ok=True)
        combined_rows = []
        for c_h in c_h_values:
            config = dict(base_config)
            config["c_h"] = float(c_h)
            run_dir = os.path.join(trajectory_dir, f"c_h_{format_scale(c_h)}")
            os.makedirs(run_dir, exist_ok=True)
            config_path = os.path.join(run_dir, "config.json")
            if not os.path.exists(config_path):
                with open(config_path, "w", encoding="utf-8") as handle:
                    json.dump(config, handle, indent=2, sort_keys=True)

            for trial in tqdm(
                probe_seeds,
                desc=f"trajectory={trajectory}, c_h={c_h}",
                unit="trial",
            ):
                raw_path = os.path.join(run_dir, f"raw_trial_{trial}.csv")
                trace_path = os.path.join(run_dir, f"sgld_trial_{trial}.npz")
                if os.path.exists(raw_path) and os.path.exists(trace_path):
                    combined_rows.extend(pd.read_csv(raw_path).to_dict("records"))
                    continue
                rows, histories = run_single_sgld(
                    config,
                    checkpoints,
                    trial,
                    model,
                    trajectory=trajectory,
                )
                for row in rows:
                    row["c_h"] = float(c_h)
                    row["Trial"] = trial
                    row["Training_Seed"] = trajectory
                combined_rows.extend(rows)
                save_trial(run_dir, trial, rows, histories)

        pd.DataFrame(combined_rows).to_csv(
            os.path.join(trajectory_dir, "sweep_raw.csv"), index=False
        )
        summary = summarize_sweep(combined_rows)
        summary.to_csv(
            os.path.join(trajectory_dir, "sweep_summary.csv"), index=False
        )
        contrast_raw, contrast_summary = paired_contrasts(combined_rows, epochs)
        contrast_raw.to_csv(
            os.path.join(trajectory_dir, "paired_contrasts_raw.csv"),
            index=False,
        )
        contrast_summary.to_csv(
            os.path.join(trajectory_dir, "paired_contrasts_summary.csv"),
            index=False,
        )
        print(f"\nTrajectory {trajectory}")
        print(summary.to_string(index=False))
        print("\nPrespecified paired contrasts:")
        print(contrast_summary.to_string(index=False))
    print(f"Output: {sweep_dir}")


if __name__ == "__main__":
    main()
