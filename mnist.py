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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='.', help="Path to the directory containing the MNIST dataset. Will download if we can't find it.")
    args = parser.parse_args()

    # Sanity check args
    if not os.path.isdir(os.path.abspath(args.data_dir)):
        print(f"{args.data_dir} is not a valid directory. Please give a path to a directory containing the MNIST dataset.")
        exit(1)

    # Download MNIST dataset
    mnist = datasets.MNIST(os.path.abspath(args.data_dir), train=True, download=True)
