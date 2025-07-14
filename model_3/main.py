"""
model 3, 
test 1 was only 6 latent dimensions did not work
test 2 was 32 latent dimensions, and 5 epochs
"""
import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Autoencoder

# Define transform to resize and normalize grayscale images


print("=== Fetching Data ===")
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0,1] and shape (1, 84, 84)
])
image_dir_path = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/Autoencoder_data"
dataset = datasets.ImageFolder(root=image_dir_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


print("=== Initializing Autoencoder Model ===")
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
print("✓ Model initialized successfully")


print("\n=== Starting Training Process ===")
# Training metrics
num_epochs = 5
outputs = []
for epoch in range(num_epochs):
    for (img, _) in dataloader:
        img = img.reshape(-1, 84*84) # -> use for Autoencoder_Linear
        recon = model(img)
        loss = criterion(recon, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    outputs.append((epoch, img, recon))

print(f"\n=== Training Complete ===")

# Save the trained model
print("\n=== Saving Model ===")
current_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(current_dir, 'trained_autoencoder.pth')
torch.save(model.state_dict(), model_save_path)
print(f"✓ Model saved to: {model_save_path}")

for k in range(0, num_epochs):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        plt.imshow(item.reshape(84, 84))
        plt.title(f'Original {i+1}')
        plt.axis('off')

    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(item.reshape(84, 84))
        plt.title(f'Reconstructed {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f'epoch_{k+1}_comparison.png')
    plt.savefig(save_path)
    plt.show()
    
