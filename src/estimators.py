import numpy as np


def apply_burn_in(sgld_history, burn_in_ratio=0.3):
    """Discard the initial burn-in portion of the MCMC chain."""
    M = len(sgld_history['L_bar_m'])
    start = int(M * burn_in_ratio)
    burned = {}
    for key, values in sgld_history.items():
        try:
            burned[key] = values[start:] if len(values) == M else values
        except TypeError:
            burned[key] = values
    return burned


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


def compute_llc_raw_variance(sgld_history, config, loss_key='L_bar_m'):
    """Raw variance-based estimator: t^2 * Var(L)."""
    if config.get('apply_burn_in', False):
        hist_burned = apply_burn_in(sgld_history, config.get('burn_in_ratio', 0.5))
    else:
        hist_burned = sgld_history
    var_L = np.var(hist_burned[loss_key], ddof=1)
    return (config['t']) ** 2 * var_L


def compute_sive_unclipped(sgld_history, config, loss_key='L_bar_m'):
    """Return the finite-path SIVE estimate without nonnegative clipping.

    Conditional on a fixed parameter path and independent centered evaluation
    noise, this is unbiased for ``t^2`` times the sample variance of the
    corresponding noiseless path losses.  A realized value may be negative.
    """
    if config.get('apply_burn_in', False):
        hist_burned = apply_burn_in(sgld_history, config.get('burn_in_ratio', 0.5))
    else:
        hist_burned = sgld_history
    N = config['N']
    if N < 2:
        raise ValueError("SIVE requires N >= 2 replicated evaluations per state.")
    var_L = np.var(hist_burned[loss_key], ddof=1)
    mean_noise_penalty = np.mean(hist_burned['s2_m']) / N
    return (config['t'] ** 2) * (var_L - mean_noise_penalty)


def compute_sive_clipped(sgld_history, config, loss_key='L_bar_m'):
    """Return ``max(0, SIVE_unclipped)`` as an optional numerical safeguard."""
    return max(0.0, compute_sive_unclipped(sgld_history, config, loss_key=loss_key))


def compute_linear_path_decomposition(sgld_history, config):
    """Decompose SIVE into residual, linear, and covariance path terms.

    For ``R = L_bar - linear_term``, the returned values obey

    ``SIVE(L_bar) = SIVE(R) + t^2 Var(linear) + 2 t^2 Cov(R, linear)``.

    The equality is algebraic on the retained finite path because the SIVE
    within-state noise correction is unchanged by subtracting a deterministic
    linear term at each state.
    """
    required = {'L_bar_detrended_m', 'linear_term_m'}
    missing = required.difference(sgld_history)
    if missing:
        raise KeyError(f"Missing path-decomposition fields: {sorted(missing)}")

    if config.get('apply_burn_in', False):
        retained = apply_burn_in(
            sgld_history,
            config.get('burn_in_ratio', 0.5),
        )
    else:
        retained = sgld_history

    linear = np.asarray(retained['linear_term_m'], dtype=float)
    residual = np.asarray(retained['L_bar_detrended_m'], dtype=float)
    if len(linear) < 2:
        raise ValueError("At least two retained states are required.")

    t2 = config['t'] ** 2
    residual_sive = compute_sive_unclipped(
        sgld_history,
        config,
        loss_key='L_bar_detrended_m',
    )
    linear_variance = t2 * np.var(linear, ddof=1)
    cross_covariance = 2 * t2 * np.cov(residual, linear, ddof=1)[0, 1]
    original_sive = compute_sive_unclipped(sgld_history, config)
    error = original_sive - (
        residual_sive + linear_variance + cross_covariance
    )
    return {
        'residual_sive_unclipped': residual_sive,
        'linear_path_variance': linear_variance,
        'linear_residual_covariance': cross_covariance,
        'decomposition_error': error,
    }


def compute_llc_debiased_variance(sgld_history, config, clip=True,
                                  loss_key='L_bar_m'):
    """Legacy entry point; new code should use the explicit ``compute_sive_*``.

    Its default preserves the original clipped behavior so archived scripts do
    not silently change meaning.  New experiment runners call the unclipped and
    clipped functions separately.
    """
    if clip:
        return compute_sive_clipped(sgld_history, config, loss_key=loss_key)
    return compute_sive_unclipped(sgld_history, config, loss_key=loss_key)
