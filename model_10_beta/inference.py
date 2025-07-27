"""
This script is used to test the ability of a model visually, it will apply inference
on a model give the path through model_path. 

= make sure the model inportting has latent dim set to the latent dim of the saved model
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from model import ConvAutoencoder

# Path to model and data
model_path = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/model_10_beta/128_dim_models/model10beta_epochs=15_latent=128_fgweight=10.0.pth"
data_dir = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/Autoencoder_data/reconstructed_images_greyscale"

print(f"[MODEL 10 INFERENCE PROGRAM]: testing model on {model_path}")
# Define transform (must match training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

# Load dataset
print("Loading dataset...")
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

# Load model
print("Loading model...")
model = ConvAutoencoder()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Get all batches and plot each
for imgs, _ in dataloader:
    with torch.no_grad():
        recons = model(imgs)
    # Convert tensors to numpy arrays for plotting
    inputs = imgs.numpy()
    recons = recons.numpy()
    # Plot original and reconstructed images in two rows
    fig, axes = plt.subplots(2, len(inputs), figsize=(2*len(inputs), 4))
    for i in range(len(inputs)):
        axes[0, i].imshow(inputs[i][0], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recons[i][0], cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    plt.tight_layout()
    plt.show()
    choice = input("press q to quit or enter to continue")
    if choice == 'q':
        break
