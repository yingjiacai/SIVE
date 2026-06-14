# Local Learning Coefficient Experiments

This repository contains small experiments for estimating the local learning
coefficient (LLC) with localized SGLD. It includes:

- toy singular models with known oracle losses;
- MNIST MLP training trajectories;
- LLC estimators based on online, retrospective, raw variance, and debiased
  variance calculations;
- scripts for appendix-style sensitivity sweeps.

## Setup

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The MNIST scripts download data into `data/` by default. Generated checkpoints,
SGLD traces, figures, and CSV files are written to `outputs/`.

## Main Scripts

Train independent MLP trajectories:

```bash
python dnn_trajectories_run.py
```

Run the main LLC experiment on saved MLP checkpoints:

```bash
python mlp_experiment_run.py
```

Run the toy-model comparison:

```bash
python toy_experiment_run.py
```

Run appendix sensitivity sweeps:

```bash
python appendix_experiment_run.py
```

Plot a saved main-experiment result:

```bash
python visualize_experiment.py
```

## Configuration

Experiment parameters live in `experiment_settings.json`. The code derives
`t = n * beta` and `lr = base_lr / t` at runtime for the SGLD experiments.

For MLP experiments, run `dnn_trajectories_run.py` first so that checkpoint
files exist under `outputs/trajectory_*/mnist_checkpoints/`.

## Notes

The scripts are research code and are intentionally lightweight. They favor
explicit experiment loops over a larger framework so that estimator behavior is
easy to inspect and modify.
