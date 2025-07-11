import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from autoEncoder import Autoencoder
from preProcess import read_reconstructed_images


# Define transform to resize and normalize grayscale images
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0,1] and shape (1, 84, 84)
])

image_dir_path = "/Users/javierfriedman/Desktop/Research/reconstructed_images_greyscale"

# Replace 'my_data/' with your actual image directory path
dataset = datasets.ImageFolder(root=image_dir_path, transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


if __name__ == "__main__":
    read_reconstructed_images()
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
