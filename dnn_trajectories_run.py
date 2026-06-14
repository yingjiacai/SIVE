import json
import os
import random

import numpy as np
import torch
from src.models import MlpModel
from src.train_dataset import train_mnist_checkpoints


def set_seed(seed):
    """Fix all sources of randomness."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single_trial(config, num_trial):
    if config["model"] != "Mlp":
        raise ValueError(f"Unknown model: {config['model']}")

    trial_dir = f"outputs/trajectory_{num_trial}"
    checkpoint_dir = os.path.join(trial_dir, "mnist_checkpoints")
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = MlpModel(root="./data", config=config)
    checkpoints = train_mnist_checkpoints(
        model,
        maxi_epochs=config["train_epochs"],
        lr=config["train_lr"],
        root=config.get("data_root", "./data"),
        checkpoint_interval=config["checkpoint_interval"],
        output_dir=checkpoint_dir,
    )


def run_experiment(config, num_trials):
    """Train multiple independent MLP trajectories from different random seeds."""
    print(f"Config: {config}")
    print(f"Running {num_trials} independent trials of MLP training...")

    for trial in range(num_trials):
        print()
        print(f"Trial {trial + 1}/{num_trials}")
        set_seed(trial)
        run_single_trial(config, trial)
        print(f"Trial {trial + 1}/{num_trials} finished")

    print("All trials finished")


if __name__ == "__main__":
    experiments_list = ["mlp_trajectories"]

    experiment_settings = json.load(open("experiment_settings.json"))

    for experiment_name in experiments_list:
        print()
        print(f"Running {experiment_name}...")

        config = experiment_settings[experiment_name]

        if "device" not in config:
            config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        num_trials = config.get("num_trials", 5)
        run_experiment(config, num_trials)
