import torch
import math


def get_localization_radius(theta_tk, config):
    """
    Compute the localization radius h for the quadratic tether.

    Toy model:
        h is an absolute scalar.
    MLP:
        h = c_h * ||theta_tk||_2 / sqrt(d), matching the paper's
        scaling h proportional to ||theta_{t_k}|| / sqrt(d).
    """
    h = config['h']
    if config['model'] == "Mlp":
        d = theta_tk.numel()
        rms_norm = torch.norm(theta_tk).item() / math.sqrt(d)
        h = h * rms_norm
        h = max(h, config.get('min_h', 1e-8))
    return h


def run_localized_sgld(model, init_theta, config):
    """
    Run localized SGLD sampling and record the trajectory.

    Implements Algorithm 1 from the paper. At each step m:
      1. Evaluate loss on N independent mini-batches.
      2. Compute gradient on one mini-batch.
      3. Apply Langevin update with localization tether.

    config keys:
        beta, n: inverse temperature and effective sample size
        h: localization bandwidth (Gaussian prior width)
        lr: learning rate (step size)
        M: number of MCMC steps
        N: number of mini-batch evaluations per step
    """
    beta, n = config['beta'], config['n']
    lr = config['lr']
    M, N = config['M'], config['N']

    if config['model'] == "Toy":
        theta_tk = init_theta.clone()
        theta = init_theta.clone()
    elif config['model'] == "Mlp":
        device = config['device']
        theta_tk = init_theta.detach().clone().to(device)
        theta = init_theta.detach().clone().to(device)
    else:
        raise ValueError(f"Unknown model: {config['model']}")
    h = get_localization_radius(theta_tk, config)

    sgld_history = {
        'L_bar_m': [],   # per-step empirical mean loss
        's2_m': [],      # per-step sample variance of loss
        'L_true_m': [],  # per-step true loss (nan for MLP)
    }

    for m in range(M):
        with torch.no_grad():
            noisy_loss, true_loss = model.evaluate(theta, N)
        L_true = true_loss.item()
        L_bar = noisy_loss.mean().item()
        s2 = noisy_loss.var(unbiased=True).item() if N > 1 else 0.0

        sgld_history['s2_m'].append(s2)
        sgld_history['L_true_m'].append(L_true)
        sgld_history['L_bar_m'].append(L_bar)

        grad = model.get_gradient(theta)

        # Localization gradient: (theta - theta_tk) / h^2
        loc_grad = (theta - theta_tk) / (h ** 2)

        # Langevin update (Algorithm 1, line 4)
        langevin_noise = torch.randn_like(theta)
        theta = theta - lr * n * beta * grad - lr * loc_grad \
                + math.sqrt(2 * lr) * langevin_noise

    return sgld_history
