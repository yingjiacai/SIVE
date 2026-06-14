import numpy as np


def apply_burn_in(sgld_history, burn_in_ratio=0.3):
    """Discard the initial burn-in portion of the MCMC chain."""
    M = len(sgld_history['L_bar_m'])
    start = int(M * burn_in_ratio)
    return {
        'L_bar_m': sgld_history['L_bar_m'][start:],
        's2_m': sgld_history['s2_m'][start:],
        'L_true_m': sgld_history['L_true_m'][start:]
    }


def compute_llc_oracle_mean(sgld_history, config, true_L0):
    """Oracle estimator using known true L0 (only for toy models)."""
    if config.get('apply_burn_in', False):
        hist_burned = apply_burn_in(sgld_history, config.get('burn_in_ratio', 0.5))
    else:
        hist_burned = sgld_history
    L_true = np.mean(hist_burned['L_true_m'])
    return config['t'] * (L_true - true_L0)


def compute_llc_naive_mean(sgld_history, config):
    """Naive estimator: t * (mean(L) - min(L))."""
    if config.get('apply_burn_in', False):
        hist_burned = apply_burn_in(sgld_history, config.get('burn_in_ratio', 0.5))
    else:
        hist_burned = sgld_history
    mean_L = np.mean(hist_burned['L_bar_m'])
    empirical_min = np.min(hist_burned['L_bar_m'])
    return config['t'] * (mean_L - empirical_min)


def compute_llc_naive_mean_specific_L(sgld_history, config, L=None):
    """Naive estimator with a user-specified baseline L (retrospective)."""
    if L is None:
        raise ValueError("A baseline loss L must be provided.")
    if config.get('apply_burn_in', False):
        hist_burned = apply_burn_in(sgld_history, config.get('burn_in_ratio', 0.5))
    else:
        hist_burned = sgld_history
    mean_L = np.mean(hist_burned['L_bar_m'])
    return config['t'] * (mean_L - L)


def compute_llc_raw_variance(sgld_history, config):
    """Raw variance-based estimator: t^2 * Var(L)."""
    if config.get('apply_burn_in', False):
        hist_burned = apply_burn_in(sgld_history, config.get('burn_in_ratio', 0.5))
    else:
        hist_burned = sgld_history
    var_L = np.var(hist_burned['L_bar_m'], ddof=1)
    return (config['t']) ** 2 * var_L


def compute_llc_debiased_variance(sgld_history, config):
    """Debiased variance estimator: t^2 * (Var(L) - mean(s2)/N), clamped at 0."""
    if config.get('apply_burn_in', False):
        hist_burned = apply_burn_in(sgld_history, config.get('burn_in_ratio', 0.5))
    else:
        hist_burned = sgld_history
    N = config['N']
    var_L = np.var(hist_burned['L_bar_m'], ddof=1)
    mean_noise_penalty = np.mean(hist_burned['s2_m']) / N
    lambda_hat = ((config['t']) ** 2) * (var_L - mean_noise_penalty)
    return max(0.0, lambda_hat)
