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
    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=False
    )
    test_dataset = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=torchvision.transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )
    os.makedirs(output_dir, exist_ok=True)
    model = probe_model.model
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    checkpoints = {}

    def save_ckpt(epoch, train_loss, test_loss):
        theta = parameters_to_vector(model.parameters()).detach().cpu().clone()
        path = os.path.join(output_dir, f"epoch_{epoch}.pt")
        temporary = f"{path}.tmp"
        torch.save({
            "epoch": epoch,
            "theta": theta,
            "train_loss": train_loss,
            "test_loss": test_loss
        }, temporary)
        os.replace(temporary, path)
        checkpoints[epoch] = path
        print(f"[Checkpoint] epoch={epoch}, path={path}")

    def evaluate_loss(data_loader):
        """Evaluate the loss of the current checkpoint parameters."""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(probe_model.device)
                y = y.to(probe_model.device)
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
        return total_loss / total_samples

    # Epoch 0 is a real evaluated checkpoint, not a placeholder zero.
    save_ckpt(0, evaluate_loss(train_eval_loader), evaluate_loss(test_loader))

    for epoch in range(1, maxi_epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(probe_model.device)
            y = y.to(probe_model.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

        # Evaluate both datasets at the saved endpoint parameters.  The old
        # logger mixed an in-epoch running training loss with endpoint test loss.
        epoch_train_loss = evaluate_loss(train_eval_loader)
        epoch_test_loss = evaluate_loss(test_loader)

        print(f"[Epoch {epoch}] Train Loss: {epoch_train_loss} Test Loss: {epoch_test_loss}")

        if epoch % checkpoint_interval == 0:
            save_ckpt(epoch, epoch_train_loss, epoch_test_loss)

    return checkpoints
