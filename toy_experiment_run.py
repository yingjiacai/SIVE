import torch
import numpy as np
import random
import json
from tqdm import tqdm
from src.models import SingularToyModel
from src.sampler import run_localized_sgld
from src.estimators import (
    compute_llc_oracle_mean,
    compute_llc_naive_mean,
    compute_llc_raw_variance,
    compute_llc_debiased_variance
)


def set_seed(seed):
    """Fix all sources of randomness."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single_trial(config):
    """Run one SGLD trial and return LLC estimates from all estimators."""
    if config['model'] == "Toy":
        model = SingularToyModel(L0=config['L0'], noise_std=config['eval_noisy'], multiplicity=config['multiplicity'])
        init_theta = torch.tensor(config['init_theta'])
    else:
        raise ValueError(f"Unknown model: {config['model']}")

    sgld_history = run_localized_sgld(model, init_theta, config)

    return {
        'oracle': compute_llc_oracle_mean(sgld_history, config, model.L0),
        'naive': compute_llc_naive_mean(sgld_history, config),
        'raw_var': compute_llc_raw_variance(sgld_history, config),
        'ours': compute_llc_debiased_variance(sgld_history, config)
    }


def run_experiment(config, num_trials):
    """Run multiple trials and print a summary table with mean +/- std."""
    results = {
        'oracle': [],
        'naive': [],
        'raw_var': [],
        'ours': []
    }
    print(f"Config: {config}")
    print(f"Running {num_trials} independent trials of Localized SGLD...")

    for trial in tqdm(range(num_trials), desc="  Trials", unit="trial"):
        set_seed(trial)
        trial_result = run_single_trial(config)
        for key in results:
            results[key].append(trial_result[key])

    stats = {}
    for key in results:
        stats[key] = {
            'mean': np.mean(results[key]),
            'std': np.std(results[key], ddof=1)
        }

    print("=" * 75)
    print(f"Table: Comparative analysis on noisy toy model (Over {num_trials} trials)")
    print("-" * 75)
    print(f"1.  Oracle Mean-based        : {stats['oracle']['mean']:.4f} +/- {stats['oracle']['std']:.4f}")
    print(f"2.  Naive Mean-based         : {stats['naive']['mean']:.4f} +/- {stats['naive']['std']:.4f}")
    print(f"3.  Raw Variance-based       : {stats['raw_var']['mean']:.4f} +/- {stats['raw_var']['std']:.4f}")
    print(f"4.  Debiased Variance        : {stats['ours']['mean']:.4f} +/- {stats['ours']['std']:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    # Example config (see experiment_settings.json for actual settings):
    #
    # config = {
    #     'beta': 1.0,               # inverse temperature
    #     'n': 10000,                # effective sample size
    #     'h': 2.0,                  # localization bandwidth
    #     'base_lr': 0.03,           # SGLD base learning rate (lr = base_lr / t)
    #     'M': 100000,               # MCMC steps
    #     'N': 64,                   # mini-batch evaluations per step
    #     'apply_burn_in': True,     # whether to apply burn-in
    #     'burn_in_ratio': 0.3,      # burn-in fraction
    #     'L0': 100,                 # true loss at the valley bottom
    #     'eval_noisy': 0.002,       # evaluation noise std
    #     'multiplicity': 2,         # 1 = non-degenerate, >1 = degenerate
    #     'init_theta': [0.2, 0.2]   # initial parameter
    # }

    experiments_list = ["4-1-1", "4-1-2"]
    experiment_settings = json.load(open("experiment_settings.json"))
    for experiment_name in experiments_list:
        print()
        print(f"Running {experiment_name}...")
        config = experiment_settings[experiment_name]
        config['t'] = config['n'] * config['beta']
        config['lr'] = config['base_lr'] / config['t']
        num_trials = 5
        run_experiment(config, num_trials)
