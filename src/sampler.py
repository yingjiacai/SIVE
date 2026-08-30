import torch
import math


def get_localization_radius(theta_tk, config):
    """
    Compute the physical localization radius for the quadratic tether.

    Toy model:
        h is an absolute scalar.
    MLP:
        h_k(c_h) = c_h * ||theta_tk||_2 / sqrt(d).
    """
    if config['model'] == "Mlp":
        c_h = config.get('c_h', config.get('h'))
        if c_h is None:
            raise KeyError("MLP configuration requires 'c_h'.")
        d = theta_tk.numel()
        rms_norm = torch.norm(theta_tk).item() / math.sqrt(d)
        h = c_h * rms_norm
        h = max(h, config.get('min_h', 1e-8))
        return h
    return config['h']


def run_localized_sgld(
    model,
    init_theta,
    config,
    reference_gradient=None,
    rng_streams=None,
):
    """
    Run localized SGLD sampling and record the trajectory.

    Implements Algorithm 1 from the paper. At each step m:
      1. Evaluate loss on N independent mini-batches.
      2. Compute gradient on one mini-batch.
      3. Apply Langevin update with localization tether.

    config keys:
        t: explicit inverse-temperature scale
        h: absolute localization bandwidth for toy models
        c_h: relative localization multiplier for MLP models
        lr: learning rate (step size)
        M: number of MCMC steps
        N: number of mini-batch evaluations per step

    ``rng_streams`` may contain independent ``evaluation``, ``gradient``, and
    ``langevin`` torch.Generator objects. Keeping them separate ensures that
    changing the number of evaluations does not alter the parameter path.
    """
    t = config['t']
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
    rng_streams = {} if rng_streams is None else rng_streams
    evaluation_generator = rng_streams.get('evaluation')
    gradient_generator = rng_streams.get('gradient')
    langevin_generator = rng_streams.get('langevin')

    if reference_gradient is not None:
        reference_gradient = reference_gradient.detach().clone().to(theta.device)
        if reference_gradient.shape != theta.shape:
            raise ValueError("reference_gradient must have the same shape as theta.")

    sgld_history = {
        'L_bar_m': [],   # per-step empirical mean loss
        's2_m': [],      # per-step sample variance of loss
        'L_true_m': [],  # per-step true loss (nan for MLP)
        'displacement_rms_m': [],
        'relative_displacement_m': [],
    }
    if reference_gradient is not None:
        sgld_history['linear_term_m'] = []
        sgld_history['L_bar_detrended_m'] = []

    for m in range(M):
        displacement_rms = torch.norm(theta - theta_tk).item() / math.sqrt(theta.numel())
        sgld_history['displacement_rms_m'].append(displacement_rms)
        sgld_history['relative_displacement_m'].append(displacement_rms / h)

        with torch.no_grad():
            noisy_loss, true_loss = model.evaluate(
                theta,
                N,
                generator=evaluation_generator,
            )
        L_true = true_loss.item()
        L_bar = noisy_loss.mean().item()
        s2 = noisy_loss.var(unbiased=True).item() if N > 1 else 0.0

        sgld_history['s2_m'].append(s2)
        sgld_history['L_true_m'].append(L_true)
        sgld_history['L_bar_m'].append(L_bar)

        if reference_gradient is not None:
            linear_term = torch.dot(reference_gradient, theta - theta_tk).item()
            sgld_history['linear_term_m'].append(linear_term)
            sgld_history['L_bar_detrended_m'].append(L_bar - linear_term)

        grad = model.get_gradient(theta, generator=gradient_generator)

        # Localization gradient: (theta - theta_tk) / h^2
        loc_grad = (theta - theta_tk) / (h ** 2)

        # Langevin update (Algorithm 1, line 4)
        langevin_noise = torch.randn(
            theta.shape,
            device=theta.device,
            dtype=theta.dtype,
            generator=langevin_generator,
        )
        theta = theta - lr * t * grad - lr * loc_grad \
                + math.sqrt(2 * lr) * langevin_noise

    return sgld_history
