import unittest

import numpy as np

from src.config import resolve_sgld_config
from src.estimators import (
    compute_llc_debiased_variance,
    compute_linear_path_decomposition,
    compute_sive_clipped,
    compute_sive_unclipped,
)


class SiveEstimatorTests(unittest.TestCase):
    def test_unclipped_can_be_negative(self):
        history = {
            "L_bar_m": [0.0, 0.0, 0.0],
            "s2_m": [1.0, 1.0, 1.0],
            "L_true_m": [0.0, 0.0, 0.0],
        }
        config = {"t": 1.0, "N": 2, "apply_burn_in": False}
        self.assertEqual(compute_sive_unclipped(history, config), -0.5)
        self.assertEqual(compute_sive_clipped(history, config), 0.0)
        self.assertEqual(compute_llc_debiased_variance(history, config), 0.0)

    def test_conditional_expectation_matches_fixed_path_variance(self):
        rng = np.random.default_rng(1234)
        repetitions = 50000
        losses = np.array([0.0, 0.5, 1.5, 4.0])
        sigmas = np.array([0.1, 0.4, 0.2, 0.8])
        n_evaluations = 5
        t = 3.0

        noise = rng.normal(
            scale=sigmas[None, :, None],
            size=(repetitions, len(losses), n_evaluations),
        )
        observations = losses[None, :, None] + noise
        group_means = observations.mean(axis=2)
        within_variances = observations.var(axis=2, ddof=1)
        estimates = (t ** 2) * (
            group_means.var(axis=1, ddof=1)
            - within_variances.mean(axis=1) / n_evaluations
        )
        target = (t ** 2) * losses.var(ddof=1)
        self.assertAlmostEqual(estimates.mean(), target, delta=0.02)

    def test_explicit_temperature_resolution(self):
        resolved = resolve_sgld_config({
            "model": "Toy",
            "t": 10000,
            "h": 2,
            "base_lr": 0.05,
            "M": 100,
            "N": 4,
        })
        self.assertEqual(resolved["t"], 10000.0)
        self.assertEqual(resolved["lr"], 5e-6)
        self.assertNotIn("legacy_t_source", resolved)

    def test_mlp_uses_relative_scale_multiplier(self):
        resolved = resolve_sgld_config({
            "model": "Mlp",
            "t": 10000,
            "c_h": 1.5,
            "base_lr": 0.05,
            "M": 100,
            "N": 4,
        })
        self.assertEqual(resolved["c_h"], 1.5)
        self.assertNotIn("h", resolved)

    def test_linear_path_decomposition_is_algebraically_exact(self):
        linear = np.array([0.0, 0.5, -0.25, 1.0])
        residual = np.array([1.0, 1.5, 2.5, 2.0])
        history = {
            "L_bar_m": (linear + residual).tolist(),
            "L_bar_detrended_m": residual.tolist(),
            "linear_term_m": linear.tolist(),
            "s2_m": [0.2, 0.4, 0.1, 0.3],
        }
        config = {"t": 3.0, "N": 5, "apply_burn_in": False}
        parts = compute_linear_path_decomposition(history, config)
        self.assertAlmostEqual(parts["decomposition_error"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
