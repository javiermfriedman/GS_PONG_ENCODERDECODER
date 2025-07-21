import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from model import ConvAutoencoder7

# -------- Main Training & Visualization --------
def main():
    # Path to reduced dataset of grayscale images
    data_dir = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/Autoencoder_data/reconstructed_images_greyscale_reduced"
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.Resize((84, 84)), # ensure images are 84x84
        transforms.ToTensor() # convert to tensor
    ])
    
    print("Loading dataset...")

    print("Loading dataset...")
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Initializing model 7)...")
    model = ConvAutoencoder7() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # -------- Training Loop --------
    print("Starting training...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for imgs, _ in dataloader:
            # Ensure imgs are on CPU (they always are by default)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")
    print("Training complete.")

    # -------- Reconstruct and Visualize --------
    print("Reconstructing images...")
    model.eval()
    # Get 10 images from the dataset (ignore labels)
    inputs = torch.stack([
        img if torch.is_tensor(img) else transforms.ToTensor()(img)
        for img, _ in (dataset[i] for i in range(10))
    ]) 
    with torch.no_grad():
        recons = model(inputs)

    inputs = inputs.numpy()
    recons = recons.numpy()

    print("Visualizing results...")
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        axes[0, i].imshow(inputs[i][0], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recons[i][0], cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    plt.tight_layout()
    plt.savefig("/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/model_7/recon_latent_6.png")
    # plt.show()

if __name__ == "__main__":
    main()
