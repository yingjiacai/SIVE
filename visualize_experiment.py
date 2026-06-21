import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DIR = "outputs/main_experiment_20260619_125458"
plt.rcParams.update({"font.size": 16})


def _compute_trial_stats(exp_dir):
    """Read all raw trial CSVs and return per-epoch stats + loss data.

    Returns:
        trials_by_epoch: {epoch: {'Online': [values], 'Ours': [...], 'Retrospective': [...]}}
        loss_by_epoch:   {epoch: (train_loss, test_loss)}
    """
    traj_dir = Path(exp_dir) / "trajectory_0"
    raw_files = sorted(glob.glob(str(traj_dir / "raw_trial_*.csv")))

    trials_by_epoch = {}
    loss_by_epoch = {}
    for f in raw_files:
        tdf = pd.read_csv(f)
        for _, row in tdf.iterrows():
            ep = int(row["Epoch"])
            if ep not in trials_by_epoch:
                trials_by_epoch[ep] = {"Online": [], "Ours": [], "Retrospective": []}
            trials_by_epoch[ep]["Online"].append(row["Online"])
            trials_by_epoch[ep]["Ours"].append(row["Ours"])
            trials_by_epoch[ep]["Retrospective"].append(row["Retrospective"])
            if ep not in loss_by_epoch:
                loss_by_epoch[ep] = (row["Train_Loss"], row["Test_Loss"])

    return trials_by_epoch, loss_by_epoch


def _fmt_mean_std(values):
    """Return (mean, std) for a list of values."""
    m = np.mean(values)
    s = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return m, s


def load_and_print_trial_table(exp_dir):
    """Print plain-text and LaTeX tables, then return the summary df for plotting."""
    traj_dir = Path(exp_dir) / "trajectory_0"
    csv_path = traj_dir / "experiment_results.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    trials_by_epoch, loss_by_epoch = _compute_trial_stats(exp_dir)
    n_trials = len(glob.glob(str(traj_dir / "raw_trial_*.csv")))
    epochs = sorted(trials_by_epoch.keys())

    # ── Plain-text table (skip epoch 0) ──────────────────────────
    print(f"\n  Based on {n_trials} trials from: {exp_dir}")
    header = (f"  {'Epoch':>6}  {'TrainLoss':>10}  {'TestLoss':>10}  "
              f"{'Online':>22} {'Retrospective':>22} {'Ours':>22}")
    print(header)
    print("-" * len(header))
    for ep in epochs:
        if ep == 0:
            continue
        d = trials_by_epoch[ep]
        tl, tstl = loss_by_epoch.get(ep, (float("nan"), float("nan")))
        on_m, on_s = _fmt_mean_std(d["Online"])
        ret_m, ret_s = _fmt_mean_std(d["Retrospective"])
        our_m, our_s = _fmt_mean_std(d["Ours"])
        print(f"  {ep:>6}  "
              f"{tl:>10.4f}  {tstl:>10.4f}  "
              f"{on_m:>12.4f} +/- {on_s:<5.4f}  "
              f"{ret_m:>12.4f} +/- {ret_s:<5.4f}"
              f"{our_m:>12.4f} +/- {our_s:<5.4f}  ")
    print()

    # ── LaTeX table (all epochs, including 0) ─────────────────────
    tex_lines = []
    tex_lines.append("% LaTeX table — copy into your document")
    tex_lines.append("% Columns: Epoch  TrainLoss  TestLoss  Online  Retrospective  Ours")
    for ep in epochs:
        d = trials_by_epoch[ep]
        tl, tstl = loss_by_epoch.get(ep, (float("nan"), float("nan")))
        on_m, on_s = _fmt_mean_std(d["Online"])
        ret_m, ret_s = _fmt_mean_std(d["Retrospective"])
        our_m, our_s = _fmt_mean_std(d["Ours"])
        tex_lines.append(
            f"  {ep} & {tl:.4f} & {tstl:.4f} & "
            f"${on_m:.4f} \\pm {on_s:.4f}$ & "
            f"${ret_m:.4f} \\pm {ret_s:.4f}$ & "
            f"${our_m:.4f} \\pm {our_s:.4f}$ \\\\"
        )

    tex_str = "\n".join(tex_lines)
    print("── LaTeX table ──")
    print(tex_str)
    print()

    # Also write to file
    tex_path = traj_dir / "experiment_results.tex"
    tex_path.write_text(tex_str + "\n")
    print(f"  LaTeX table saved to: {tex_path}\n")

    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    else:
        exp_dir = DIR

    df = load_and_print_trial_table(exp_dir)
    df = df[df["Epoch"] > 0]
    epoch = df["Epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 7), sharex=True)
    (ax_loss, ax_loss_log), (ax_llc, ax_llc_log) = axes

    # ---- Top-left: Loss (original) ----
    ax_loss.plot(epoch, df["Train_Loss"], color="tab:blue", label="Train Loss")
    ax_loss.plot(epoch, df["Test_Loss"], color="tab:orange", label="Test Loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    # ---- Top-right: Loss (log) ----
    ax_loss_log.plot(epoch, df["Train_Loss"], color="tab:blue", label="Train Loss")
    ax_loss_log.plot(epoch, df["Test_Loss"], color="tab:orange", label="Test Loss")
    ax_loss_log.set_ylabel("Loss (log scale)")
    ax_loss_log.set_yscale("log")
    ax_loss_log.legend(loc="upper right")
    ax_loss_log.grid(True, alpha=0.3)

    # ---- Bottom-left: LLC (original) ----
    def plot_with_shade(ax, x, mean_col, std_col, color, label):
        mean = df[mean_col]
        std = df[std_col]
        ax.plot(x, mean, color=color, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    plot_with_shade(ax_llc, epoch, "Online_mean", "Online_std", "tab:green", "Online")
    plot_with_shade(ax_llc, epoch, "Retrospective_mean", "Retrospective_std", "tab:red", "Retrospective")
    plot_with_shade(ax_llc, epoch, "Ours_mean", "Ours_std", "tab:purple", "SIVE")
    ax_llc.set_ylabel("LLC")
    ax_llc.legend(loc="upper right")
    ax_llc.grid(True, alpha=0.3)

    # ---- Bottom-right: LLC (log) ----
    plot_with_shade(ax_llc_log, epoch, "Online_mean", "Online_std", "tab:green", "Online")
    plot_with_shade(ax_llc_log, epoch, "Retrospective_mean", "Retrospective_std", "tab:red", "Retrospective")
    plot_with_shade(ax_llc_log, epoch, "Ours_mean", "Ours_std", "tab:purple", "SIVE")
    ax_llc_log.set_ylabel("LLC (log scale)")
    ax_llc_log.set_yscale("log")
    ax_llc_log.legend(loc="upper right")
    ax_llc_log.grid(True, alpha=0.3)

    # Shared x labels for bottom row
    ax_llc.set_xlabel("Epoch")
    ax_llc_log.set_xlabel("Epoch")

    plt.tight_layout()
    out_path = Path(exp_dir) / "trajectory_0" / "experiment_results.pdf"
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path} (dpi=200)")
    plt.show()