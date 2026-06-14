import os
import torch
import torchvision
from torch.nn.utils import parameters_to_vector


def train_mnist_checkpoints(probe_model, maxi_epochs=20, lr=0.1, root='./data',
                            checkpoint_interval=1, output_dir="outputs/mnist_checkpoints"):
    """Train an MLP on MNIST with SGD and save checkpoints at each interval.

    No early stopping; runs exactly maxi_epochs. Accuracy is not monitored;
    only train/test loss is tracked.
    """
    train_dataset = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=torchvision.transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    os.makedirs(output_dir, exist_ok=True)
    model = probe_model.model
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    checkpoints = {}

    def save_ckpt(epoch, train_loss, test_loss):
        theta = parameters_to_vector(model.parameters()).detach().cpu().clone()
        path = os.path.join(output_dir, f"epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "theta": theta,
            "train_loss": train_loss,
            "test_loss": test_loss
        }, path)
        checkpoints[epoch] = path
        print(f"[Checkpoint] epoch={epoch}, path={path}")

    save_ckpt(0, 0.0, 0.0)

    for epoch in range(1, maxi_epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_samples = 0

        for x, y in train_loader:
            x = x.to(probe_model.device)
            y = y.to(probe_model.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x.size(0)
            train_samples += x.size(0)

        epoch_train_loss = total_train_loss / train_samples

        model.eval()
        total_test_loss = 0.0
        test_samples = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(probe_model.device)
                y = y.to(probe_model.device)
                test_logits = model(x)
                test_loss = torch.nn.functional.cross_entropy(test_logits, y)
                total_test_loss += test_loss.item() * x.size(0)
                test_samples += x.size(0)

        epoch_test_loss = total_test_loss / test_samples

        print(f"[Epoch {epoch}] Train Loss: {epoch_train_loss} Test Loss: {epoch_test_loss}")

        if epoch % checkpoint_interval == 0:
            save_ckpt(epoch, epoch_train_loss, epoch_test_loss)

    return checkpoints
