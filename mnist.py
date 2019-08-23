"""
PyTorch MNIST

Plenty taken from https://github.com/pytorch/examples/blob/master/mnist/main.py
Some taken from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder
Some taken from https://github.com/sksq96/pytorch-vae/blob/master/vae.py
"""
# pylint: disable=no-member
import argparse
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms


MNIST_DIMS = (28, 28)
MNIST_FLAT_SIZE = MNIST_DIMS[0] * MNIST_DIMS[1]
MNIST_CLASSES = 10  # One for each digit (0 through 9)


class MnistMlpModel(nn.Module):
    """
    MLP model for MNIST. Simplest possible stuff.
    """
    def __init__(self):
        super(MnistMlpModel, self).__init__()
        self.fc1 = nn.Linear(MNIST_FLAT_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, MNIST_CLASSES)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

class MnistCnnModel(nn.Module):
    def __init__(self):
        super(MnistCnnModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, MNIST_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MnistAutoEncoderModel(nn.Module):
    def __init__(self):
        super(MnistAutoEncoderModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unflatten(nn.Module):
    def __init__(self, size):
        super(Unflatten, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(x.size(0), self.size, 1, 1)

class MnistVariationalAutoencoderModel(nn.Module):
    def __init__(self, device):
        super(MnistVariationalAutoencoderModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=8),
            Flatten()
        )
        self.fc_left = nn.Linear(in_features=32, out_features=20)
        self.fc_right = nn.Linear(in_features=32, out_features=20)
        self.collect = nn.Linear(in_features=20, out_features=8*16)
        self.decoder = nn.Sequential(
            Unflatten(8*16),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=8),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=4, stride=3),
            nn.Sigmoid()
        )
        self.device = device

    def encode(self, x):
        h = F.leaky_relu(self.encoder(x))
        return self.fc_left(h), self.fc_right(h)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.device.type.startswith("cuda"):
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.autograd.Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = self.collect(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvariance = self.encode(x)
        z = self.reparameterize(mu, logvariance)
        return self.decode(z), mu, logvariance


def train(model: nn.Module, device: torch.torch.device, train_loader: torch.utils.data.DataLoader, optimizer: optim.Optimizer, epoch: int):
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
        yhat = model(x)

        # Calculate loss score
        loss = F.nll_loss(yhat, y)

        # Back prop
        loss.backward()
        optimizer.step()

        # Maybe print some useful stuff
        if batchidx % 10 == 0:
            print(f"Epoch {epoch}: [{batchidx * len(x)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}", end='\r')
    print("")

def train_autoencoder(model: nn.Module, device: torch.torch.device, train_loader: torch.utils.data.DataLoader, optimizer: optim.Optimizer, epoch: int):
    """
    Do a single epoch of training on an autoencoder model.
    """
    model.train()
    for batchidx, (x, _) in enumerate(train_loader):
        # Send data and labels to device
        x = x.to(device)
        y = x

        # Zero the optimizer
        optimizer.zero_grad()

        # Forward pass
        yhat = model(x)

        # Calculate the loss score
        loss = F.mse_loss(yhat, y)

        # Back prop
        loss.backward()
        optimizer.step()

        # Maybe print some useful stuff
        if batchidx % 10 == 0:
            print(f"Epoch {epoch}: [{batchidx * len(x)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}", end='\r')
    print("")

def train_vae(model: nn.Module, device: torch.torch.device, train_loader: torch.utils.data.DataLoader, optimizer: optim.Optimizer, epoch: int):
    """
    Do a single epoch of training for a variational autoencoder model.
    """
    model.train()
    for batchidx, (x, _) in enumerate(train_loader):
        # Get the reconstruction, the mu, and the log of the variance
        recon, mu, logvar = model(x.to(device))

        # Compute the loss values
        binary_xentropy_loss = F.binary_cross_entropy(recon, x, size_average=False)
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = binary_xentropy_loss + kl_divergence

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Maybe print some useful stuff
        if batchidx % 10 == 0:
            print(f"Epoch {epoch}: [{batchidx * len(x)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}", end='\r')
    print("")

def test(model: nn.Module, device: torch.torch.device, test_loader: torch.utils.data.DataLoader):
    """
    Test the MNIST model.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)
            test_loss += F.nll_loss(yhat, y, reduction='sum').item()  # sum up batch loss
            pred = yhat.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

def test_autoencoder(model: nn.Module, device: torch.torch.device, test_loader: torch.utils.data.DataLoader):
    """
    Test the MNIST autoencoder models.
    """
    model.eval()
    with torch.no_grad():
        imgs = []
        for x, _ in test_loader:
            x = x.to(device)
            yhat, _mu, _logvar = model(x)
            imgs.append(yhat[0].squeeze())

        # Show a bunch of images
        nrows = 4
        ncols = 4
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, constrained_layout=False)
        for rowidx in range(nrows):
            for colidx in range(ncols):
                item = imgs[rowidx * ncols + colidx]
                axs[rowidx][colidx].imshow(item)
                axs[rowidx][colidx].axis('off')
        fig.suptitle("Samples of Autoencoder Values")
        plt.show()

# TODO: Test VAE by sampling from latent space


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
    transformations = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if args.model == 'mlp':
        # Need to flatten input
        transformations.append(transforms.Lambda(lambda x: x.view(-1)))
    train_transform = transforms.Compose(transformations)
    test_transform = transforms.Compose(list(transformations))
    mnist_train = datasets.MNIST(os.path.abspath(args.data_dir), train=True, download=True, transform=train_transform)
    mnist_test = datasets.MNIST(os.path.abspath(args.data_dir), train=False, download=True, transform=test_transform)

    # Set up the data loaders
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=True)

    # Create the model
    if args.model == 'mlp':
        model = MnistMlpModel()
    elif args.model == 'cnn':
        model = MnistCnnModel()
    elif args.model == 'autoencoder':
        model = MnistAutoEncoderModel()
    elif args.model == 'vae':
        model = MnistVariationalAutoencoderModel(device)
    else:
        raise NotImplementedError(f"{args.model} is not yet implemented. Sorry :(")

    # Set the model's tensors to the right type
    model = model.to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model; test after every epoch
    for epoch in range(1, args.n_epochs + 1):
        if args.model == 'mlp' or args.model == 'cnn':
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
        elif args.model == 'autoencoder':
            train_autoencoder(model, device, train_loader, optimizer, epoch)
            test_autoencoder(model, device, test_loader)
        elif args.model == 'vae':
            train_vae(model, device, train_loader, optimizer, epoch)
            test_autoencoder(model, device, test_loader)
