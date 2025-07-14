"""
this
Autoencoder training script for image reconstruction using PyTorch.
This script trains an autoencoder on grayscale images and saves the model with training metrics.
this is the preliminary model for the encoder doesn't work well
only used linear layer
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
num_epochs = 5
outputs = []

# Training metrics
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    epoch_loss = 0
    epoch_accuracy = 0
    num_batches = 0
    
    for batch_idx, (img, _) in enumerate(dataloader):
       
        img = img.reshape(-1, 84*84)  # Reshape img to (batch_size, 84*84)
        recon = model(img)
        loss = criterion(recon, img)
        
        # Calculate accuracy (how well reconstruction matches original)
        with torch.no_grad():
            mse = torch.mean((recon - img) ** 2)
            accuracy = 1.0 - mse.item()  # Higher is better
        
        optimizer.zero_grad()  # Clear old gradients.
        loss.backward()  # Compute gradients.
        optimizer.step()  # Update weights.
        
        epoch_loss += loss.item()
        epoch_accuracy += accuracy
        num_batches += 1
        
        # Print batch metrics every 10 batches
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
    
    # Calculate epoch averages
    avg_loss = epoch_loss / num_batches
    avg_accuracy = epoch_accuracy / num_batches
    
    train_losses.append(avg_loss)
    train_accuracies.append(avg_accuracy)
    
    print(f"  ✓ Epoch {epoch+1} Summary:")
    print(f"    - Average Loss: {avg_loss:.4f}")
    print(f"    - Average Accuracy: {avg_accuracy:.4f}")
    print(f"    - Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    outputs.append((epoch, img, recon))

print(f"\n=== Training Complete ===")
print(f"Final Loss: {train_losses[-1]:.4f}")
print(f"Final Accuracy: {train_accuracies[-1]:.4f}")
print(f"Best Loss: {min(train_losses):.4f} (Epoch {train_losses.index(min(train_losses))+1})")
print(f"Best Accuracy: {max(train_accuracies):.4f} (Epoch {train_accuracies.index(max(train_accuracies))+1})")

# Save the trained model
print("\n=== Saving Model ===")
current_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(current_dir, 'trained_autoencoder.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'num_epochs': num_epochs,
    'final_loss': train_losses[-1],
    'final_accuracy': train_accuracies[-1]
}, model_save_path)
print(f"✓ Model saved to: {model_save_path}")

# plot the reconstructed images and compare to the oringaol images
for k in range(0, num_epochs):
    plt.figure(figsize=(12, 3))
    plt.gray()

    epoch_loss = train_losses[k]
    epoch_accuracy = train_accuracies[k]
    lr = optimizer.param_groups[0]['lr']
    
    # Create title with metrics
    title = f'Epoch {k+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, LR: {lr:.6f}'
    plt.suptitle(title, fontsize=12, fontweight='bold')
    
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    
    # Loop through the original images and display them.
    for i, item in enumerate(imgs):
        # Display a maximum of 9 images.
        if i >= 9: break
        # Create a subplot for each image in a 2x9 grid.
        plt.subplot(2, 9, i+1)
        # Reshape the image to its original 2D shape (84x84) and display it.
        item = item.reshape(-1, 84,84) 
        plt.imshow(item[0])
        plt.title(f'Original {i+1}', fontsize=8)
        plt.axis('off')
            
    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1) # row_length + i + 1
        item = item.reshape(-1, 84,84) # -> use for Autoencoder_Linear
        plt.imshow(item[0])
        plt.title(f'Reconstructed {i+1}', fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f'epoch_{k+1}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved chart to: {save_path}")
    
    plt.show()
