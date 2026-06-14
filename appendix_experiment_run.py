import torch
import numpy as np
import random
import json
from tqdm import tqdm
from src.models import SingularToyModel
from src.sampler import run_localized_sgld
from src.estimators import (
    compute_llc_oracle_mean,
    compute_llc_debiased_variance
)


def set_seed(seed):
    """Fix all sources of randomness."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single_trial(config):
    """Run one SGLD trial and return oracle, debiased, and their difference."""
    if config['model'] == "Toy":
        model = SingularToyModel(L0=config['L0'], noise_std=config['eval_noisy'], multiplicity=config['multiplicity'])
        init_theta = torch.tensor(config['init_theta'])
    else:
        raise ValueError(f"Unknown model: {config['model']}")

    sgld_history = run_localized_sgld(model, init_theta, config)

    oracle = compute_llc_oracle_mean(sgld_history, config, model.L0)
    ours = compute_llc_debiased_variance(sgld_history, config)
    return {
        'oracle': oracle,
        'ours': ours,
        'diff': oracle - ours
    }


def run_experiment(config, num_trials):
    """Run multiple trials and return per-estimator mean +/- std stats."""
    results = {
        'oracle': [],
        'ours': [],
        'diff': []
    }

    for trial in tqdm(range(num_trials), desc="  Trials", unit="trial", leave=False):
        set_seed(trial)
        trial_result = run_single_trial(config)
        for key in results:
            results[key].append(trial_result[key])

    return {
        key: {
            'mean': np.mean(results[key]),
            'std': np.std(results[key], ddof=1)
        }
        for key in results
    }


if __name__ == "__main__":
    print("Variable choices: h, lr1, lr2, N")
    print("  lr = base_lr (will be divided by t)")
    print("  lr1 for appendix_c_2, lr2 for appendix_d")
    variable_name = input("variable_name: ")

    if variable_name == "h":
        print("Running Appendix C.1: Localization sensitivity to h...")

        experiment_settings = json.load(open("experiment_settings.json"))
        config = experiment_settings["appendix_c_1"]
        config['t'] = config['n'] * config['beta']
        config['lr'] = config['base_lr'] / config['t']

        print(f"config: {config}")
        h_list = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
        print(f"h_list: {h_list}")
        num_trials = 5

        print()
        print("-" * 65)
        print(f"{'h':>8}  {'Oracle Mean-based':>27}  {'Debiased Variance':>27}")
        print("-" * 65)

        for h in h_list:
            config['h'] = h
            stats = run_experiment(config, num_trials)
            print(f"{h:>8.2f}  {stats['oracle']['mean']:>18.4f} +/- {stats['oracle']['std']:<5.4f}"
                  f"  {stats['ours']['mean']:>18.4f} +/- {stats['ours']['std']:<5.4f}")

        print("-" * 65)

    if variable_name == "lr2" or variable_name == "lr1":
        print("Running Appendix C.3: Localization sensitivity to lr...")
        experiment_settings = json.load(open("experiment_settings.json"))

        config = experiment_settings["appendix_c_1"]
        if variable_name == "lr2":
            config['multiplicity'] = 2
            lr_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
        elif variable_name == "lr1":
            config['multiplicity'] = 1
            lr_list = [0.05, 0.1]
        else:
            raise ValueError(f"Unknown variable name: {variable_name}")

        config['t'] = config['n'] * config['beta']
        config['lr'] = config['base_lr'] / config['t']

        print(f"config: {config}")
        print(f"lr_list: {lr_list}")
        num_trials = 5

        print()
        print("-" * 65)
        print(f"{'base_lr':>8}  {'Oracle Mean-based':>27}  {'Debiased Variance':>27}")
        print("-" * 65)

        for lr in lr_list:
            config['base_lr'] = lr
            config['lr'] = config['base_lr'] / config['t']
            stats = run_experiment(config, num_trials)
            print(f"{lr:>8.5f}  {stats['oracle']['mean']:>18.4f} +/- {stats['oracle']['std']:<5.4f}"
                  f"  {stats['ours']['mean']:>18.4f} +/- {stats['ours']['std']:<5.4f}")

        print("-" * 65)

    if variable_name == "N":
        N_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        print("Running Appendix C.4: Localization sensitivity to N...")

        experiment_settings = json.load(open("experiment_settings.json"))
        config = experiment_settings["appendix_c_1"]
        config['t'] = config['n'] * config['beta']
        config['lr'] = config['base_lr'] / config['t']

        print(f"config: {config}")
        print(f"N_list: {N_list}")
        num_trials = 5

        print()
        print("-" * 95)
        print(f"{'N':>8}  {'Oracle Mean-based':>27}  {'Debiased Variance':>27}  {'Delta (Oracle - Ours)':>27}")
        print("-" * 95)

        for N in N_list:
            config['N'] = N
            stats = run_experiment(config, num_trials)
            print(f"{N:>8}  {stats['oracle']['mean']:>18.4f} +/- {stats['oracle']['std']:<5.4f}"
                  f"  {stats['ours']['mean']:>18.4f} +/- {stats['ours']['std']:<5.4f}"
                  f"  {stats['diff']['mean']:>18.4f} +/- {stats['diff']['std']:<5.4f}")

        print("-" * 95)

    if variable_name not in {"h", "lr1", "lr2", "N"}:
        raise ValueError(f"Unknown variable name: {variable_name}")
