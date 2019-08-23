"""
PyTorch MNIST
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets


MNIST_DIMS = (28, 28)
MNIST_FLAT_SIZE = MNIST_DIMS[0] * MNIST_DIMS[1]
MNIST_CLASSES = 10  # One for each digit (0 through 9)


class MnistMlpModel(nn.Module):
    """
    MLP model for MNIST. Simplest possible stuff.
    """
    def __init__(self):
        super(MnistMlpModel, self).__init__()
        self.fc1 = nn.Linear(MNIST_FLAT_SIZE, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, MNIST_CLASSES)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x


def train(model: nn.Module, device: torch.torch.device, train_loader: torch.utils.data.DataLoader, optimizer: None, epoch: int):
    """
    Do a single epoch of training on the model.
    """
    model.train()
    for batchidx, (x, y) in enumerate(train_loader):
        # Send data and labels to device
        x = x.to(device)
        y = y.to(device)

        # Zero the optimizer
        optimizer.zero_grad()

        # Forward pass
        yhat =model(x)

        # Calculate loss score
        loss = F.nll_loss(yhat, y)

        # Back prop
        loss.backward()
        optimizer.stop()

        # Maybe print some useful stuff
        if batchidx % 10 == 0:
            print(f"Epoch {epoch}: [{batchidx * len(x)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--cuda', action='store_true', help="If given, will attempt to use CUDA on GPU.")
    parser.add_argument('--data-dir', type=str, default='.', help="Path to the directory containing the MNIST dataset. Will download if we can't find it.")
    parser.add_argument('--model', choices=('mlp', 'cnn', 'autoencoder', 'vae'), default='mlp', help="Model to use.")
    parser.add_argument('--n-epochs', type=int, default=1, help="Number of epochs of training.")
    args = parser.parse_args()

    # Sanity check args
    if not os.path.isdir(os.path.abspath(args.data_dir)):
        print(f"{args.data_dir} is not a valid directory. Please give a path to a directory containing the MNIST dataset.")
        exit(1)
    elif args.batch_size <= 0:
        print(f"Must have batch-size > 0, but is {args.batch_size}")
        exit(2)
    elif args.n_epochs <= 0:
        print(f"Must have n-epochs > 0, but is {args.n_epochs}")
        exit(3)

    # Determine device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Download MNIST dataset
    mnist_train = datasets.MNIST(os.path.abspath(args.data_dir), train=True, download=True)
    mnist_test = datasets.MNIST(os.path.abspath(args.data_dir), train=False, download=True)

    # Set up the data loaders
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=True)

    # Create the model
    if args.model == 'mlp':
        model = MnistMlpModel()
    else:
        raise NotImplementedError(f"{args.model} is not yet implemented. Sorry :(")

    # Set the model's tensors to the right type
    model = model.to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model; test after every epoch
    for epoch in range(1, args.n_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model)
