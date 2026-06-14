import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"font.size": 16})

csv_path = Path(__file__).parent / "outputs/main_experiment_20260520_015120/trajectory_0/experiment_results.csv"
df = pd.read_csv(csv_path)
df = df[df["Epoch"] > 0]

epoch = df["Epoch"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# ---- Top: Loss curves ----
ax1.plot(epoch, df["Train_Loss"], color="tab:blue", label="Train Loss")
ax1.plot(epoch, df["Test_Loss"], color="tab:orange", label="Test Loss")
ax1.set_ylabel("Loss")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# ---- Bottom: Mean metrics with shaded std ----
def plot_with_shade(ax, x, mean_col, std_col, color, label):
    mean = df[mean_col]
    std = df[std_col]
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

plot_with_shade(ax2, epoch, "Online_mean", "Online_std", "tab:green", "Online")
plot_with_shade(ax2, epoch, "Retrospective_mean", "Retrospective_std", "tab:red", "Retrospective")
plot_with_shade(ax2, epoch, "Ours_mean", "Ours_std", "tab:purple", "SIVE")

ax2.set_xlabel("Epoch")
ax2.set_ylabel("LLC")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = csv_path.parent / "experiment_results.pdf"
fig.savefig(out_path)
print(f"Saved to {out_path}")
plt.show()