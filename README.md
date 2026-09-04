# SIVE finite-scale local-probe experiments

This repository contains the experiment code for the Shift-Invariant Variance
Estimator (SIVE). The implemented statistic is a finite-path, finite-scale
local loss-fluctuation probe. It should not be identified with an intrinsic
LLC/RLCT at an arbitrary non-stationary checkpoint without the additional
stationarity, sampling, and low-temperature limits described in the paper.

The repository includes:

- controlled regular and singular toy losses with privileged oracle baselines;
- end-to-end Online, Retrospective, raw-variance, and SIVE comparisons;
- MNIST MLP training trajectories and localized-SGLD checkpoint probes;
- paired multi-scale and checkpoint-gradient diagnostics;
- a continuous Gaussian-localized calibration for `K(u,v) = u^2 v^2`.

## Setup

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

MNIST is downloaded into `data/` by default. Checkpoints, raw SGLD traces,
manifests, figures, and summary tables are written below `outputs/`.

## Configuration semantics

Experiment parameters live in `experiment_settings.json`.

- `t` is the explicit inverse-temperature scale used by the localized law.
- `h` is the absolute Gaussian width for a toy model.
- `c_h` is the MLP's relative multiplier; the physical checkpoint width is
  `h_k(c_h) = c_h * ||theta_checkpoint|| / sqrt(d)`.
- `base_lr` defines the SGLD step as `lr = base_lr / t`.
- `N` is the number of independent evaluation groups per sampled state.
- `full_gradient_batch_size` controls the memory-bounded traversal used to
  compute the exact full-training-set checkpoint gradient.
- `record_linear_detrending` records the loss after subtracting the fixed
  checkpoint linearization `g_hat^T(theta - theta_checkpoint)`.

New configurations should specify `t` directly. The code can still read the
legacy pair `n` and `beta` solely for archived runs.

## Estimator outputs

New runs report both variants explicitly:

- `SIVE_unclipped` is the primary statistic analyzed in the paper. A finite
  realization can be negative.
- `SIVE_clipped = max(0, SIVE_unclipped)` is an optional numerical safeguard
  and is not conditionally unbiased.

For DNN runs, the output also includes `Raw_Variance`,
`Full_Gradient_Norm_Sq`, `Scale_Multiplier_c_h`, `Physical_h`, the tight-tether
diagnostic `Slope_Proxy = t^2 Physical_h^2 Full_Gradient_Norm_Sq`, and the
corresponding residual SIVE value. `Linear_Path_Variance` and
`Linear_Residual_Covariance` complete the exact finite-path variance
decomposition. These are diagnostics, not a claim that the nonlinear slope
contribution has been removed exactly.

Gradient-batch indices, Langevin Gaussian draws, and evaluation-batch indices
use independent deterministic random streams. Their seeds depend on training
trajectory, probe trial, and checkpoint, but not on `c_h`; the scale audit
therefore uses common random numbers across multipliers. The seed is the
explicit decimal encoding `1_000_000*t + 1_000*k + 10*s + stream_id`, where
`t`, `k`, and `s` are trajectory, checkpoint, and probe seed, and the three
stream IDs are 0, 1, and 2.
The five probe seeds themselves are the explicit `probe_seeds` list in
`experiment_settings.json`.

## Reproducing the workflow

### Five-trajectory DNN run

The complete run is one shell command:

```bash
bash run_dnn_pipeline.sh
```

It trains seeds 1--5, runs the five-probe main comparison for each trajectory,
then runs the configured multi-`c_h` audit. Results go to a new directory such
as `outputs/dnn_5traj_YYYYMMDD_HHMMSS`; older outputs are not used or changed.
The seed list is the `SEEDS` line near the top of the shell script.

To continue an interrupted run, pass the same output directory again:

```bash
bash run_dnn_pipeline.sh outputs/dnn_5traj_YYYYMMDD_HHMMSS
```

Completed training trajectories and probe trials are skipped. Each training
trajectory keeps its own `experiment_results.csv`, `sweep_summary.csv`, and
paired-contrast table, so cross-trajectory reporting can be assembled later
without pooling the 25 probe trials as if they were 25 trained networks.

### Individual entry points

Train one MLP trajectory and evaluate real train/test losses at every saved
checkpoint, including epoch 0:

```bash
python dnn_trajectories_run.py
```

Run the main end-to-end comparison over all saved checkpoints:

```bash
python mlp_experiment_run.py
```

Run the paired multi-scale audit at the checkpoint epochs configured under
`dnn_h_sweep`. It also writes the prespecified early--middle drop and
late--middle rebound contrasts:

```bash
python mlp_probe.py
```

Run the toy comparison or appendix sweeps:

```bash
python toy_experiment_run.py
python appendix_experiment_run.py
```

Reproduce the continuous finite-temperature reference used for the shifted
singular toy experiment, without running SGLD:

```bash
python calibrate_toy_reference.py --t 10000 --h 2 --center-u 0.2 --center-v 0.2
```

Plot a completed main experiment:

```bash
python visualize_experiment.py outputs/main_experiment_YYYYMMDD_HHMMSS
```

The probe panel uses a symmetric-log scale so that legitimate negative
unclipped estimates remain visible.

## Reproducibility records

Long-running scripts write a `run_manifest.json` containing the resolved
configuration, seeds, command, Python/platform information, Git state, and
SHA-256 hashes of the Python and JSON source files. Raw per-trial CSV and NPZ
files are retained alongside aggregate summaries. No DNN result is generated
or implied merely by changing these scripts; the experiment must be rerun.
