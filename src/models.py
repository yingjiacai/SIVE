import torch
import torchvision


class MlpModel:
    def __init__(self, root='./data', config=None):
        self.model = self.MlpNet()
        self.device = torch.device(config['device'])
        self.model.to(self.device)
        self.eval_batch_size = config.get('eval_batch_size', 64)
        self.grad_batch_size = config.get('grad_batch_size', 64)

        # Load full MNIST training set into memory for fast mini-batch sampling
        train_dataset = torchvision.datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=len(train_dataset),
            shuffle=False
        )
        self.X, self.Y = next(iter(loader))
        self.X = self.X.to(self.device)
        self.Y = self.Y.to(self.device)
        self.num_samples = self.X.shape[0]

        # Cache parameter structure for converting flat theta to parameter dicts
        self.param_names = []
        self.param_shapes = []
        self.param_numels = []
        for name, p in self.model.named_parameters():
            self.param_names.append(name)
            self.param_shapes.append(p.shape)
            self.param_numels.append(p.numel())

    def _theta_to_params(self, theta):
        """Convert a flat parameter vector into a parameter dict for functional_call.

        We avoid torch.nn.utils.vector_to_parameters because it mutates in-place
        and breaks the autograd graph.
        """
        params = {}
        pointer = 0
        for name, shape, numel in zip(self.param_names, self.param_shapes, self.param_numels):
            params[name] = theta[pointer:pointer + numel].view(shape)
            pointer += numel
        return params

    def evaluate(self, theta, N=64):
        """Evaluate the model on N independent mini-batches.

        Returns:
            group_losses: (N,) tensor of per-mini-batch mean cross-entropy.
                sampler.py computes mean and variance over these N values.
            true_loss: nan (no oracle loss for MLP/MNIST experiments).
        """
        theta = theta.to(self.device)
        params = self._theta_to_params(theta)
        B = self.eval_batch_size
        indices = torch.randint(
            low=0,
            high=self.num_samples,
            size=(N, B),
            device=self.device
        )
        flat_indices = indices.reshape(-1)
        X = self.X[flat_indices]
        Y = self.Y[flat_indices]
        logits = torch.func.functional_call(self.model, params, (X,))
        per_sample_loss = torch.nn.functional.cross_entropy(
            logits,
            Y,
            reduction='none'
        )
        per_sample_loss = per_sample_loss.view(N, B)
        group_losses = per_sample_loss.mean(dim=1)
        true_loss = torch.tensor(
            float('nan'),
            device=self.device,
            dtype=group_losses.dtype
        )
        return group_losses, true_loss

    def get_gradient(self, theta):
        """Compute the gradient of empirical loss w.r.t. theta on a single mini-batch."""
        theta = theta.detach().clone().to(self.device).requires_grad_(True)
        params = self._theta_to_params(theta)
        indices = torch.randint(0, self.num_samples, (self.grad_batch_size,), device=self.device)
        X = self.X[indices]
        Y = self.Y[indices]
        logits = torch.func.functional_call(self.model, params, (X,))
        loss = torch.nn.functional.cross_entropy(logits, Y, reduction='mean')
        grad = torch.autograd.grad(loss, theta)[0]
        return grad.detach()

    class MlpNet(torch.nn.Module):
        def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            x = torch.relu(x)
            x = self.fc3(x)
            return x

class SingularToyModel:
    def __init__(self, L0=10.0, noise_std=0.1, multiplicity=1):
        """
        L0: true loss value at the bottom of the valley (unknown in real settings).
        noise_std: standard deviation of simulated mini-batch evaluation noise.
        multiplicity: 1 = non-degenerate, >1 = degenerate (singular) case.
        """
        self.L0 = L0
        self.noise_std = noise_std
        self.multiplicity = multiplicity

    def evaluate(self, theta, N=1):
        """Compute loss with simulated mini-batch noise.

        theta: (2,) tensor [u, v].
        Returns: (noisy_losses, true_loss) where noisy_losses has shape (N,).
        """
        if self.multiplicity == 1:
            true_loss = self.L0 + (theta[0] ** 2)
        else:
            true_loss = self.L0 + (theta[0] ** 2) * (theta[1] ** 2)

        noise = torch.randn(N, device=theta.device, dtype=theta.dtype) * self.noise_std
        noisy_loss = true_loss + noise
        return noisy_loss, true_loss

    def get_gradient(self, theta, noisy=False):
        """Compute gradient of the true loss w.r.t. theta via autograd.

        noisy=True for stochastic gradient is not yet implemented.
        """
        if noisy:
            raise NotImplementedError("Noisy gradient is not implemented yet")

        theta_with_grad = theta.detach().clone().requires_grad_(True)
        loss = self.evaluate(theta_with_grad)[1]
        loss.backward()
        return theta_with_grad.grad.detach()
