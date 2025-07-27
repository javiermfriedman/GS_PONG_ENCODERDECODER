# vae_main.py
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from vae_model import VAE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # It's good practice to make this path easily configurable
    data_dir = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/Autoencoder_data/reconstructed_images_greyscale_reduced"

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # --- UPDATED MODEL INITIALIZATION ---
    # Instantiate the VAE with a beta value (e.g., 0.5) to balance the loss
    model = VAE(latent_dim=32, beta=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_losses = []
    num_epochs = 50 # You may want to train for more epochs

    print(f"Starting training on {device} with beta={model.beta}...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            # The loss function in the model now handles beta automatically
            loss, _, _ = model.loss_function(recon, imgs, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                recon, mu, logvar = model(imgs)
                loss, _, _ = model.loss_function(recon, imgs, mu, logvar)
                val_loss += loss.item()

        train_losses.append(train_loss / len(train_loader.dataset))
        val_losses.append(val_loss / len(val_loader.dataset))

        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

    #save model
    model_path = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/model_7/vae_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'VAE Training Loss (β={model.beta})')
    plt.show()
    plt.savefig("/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/model_7/vae_loss_plot.png")

    # Visualize reconstructions
    print("Generating reconstruction comparison...")
    model.eval()
    # Ensure you have at least 10 images in the validation set
    num_samples = min(10, len(val_dataset))
    inputs = torch.stack([val_dataset[i][0] for i in range(num_samples)]).to(device)
    with torch.no_grad():
        recons, _, _ = model(inputs)

    inputs = inputs.cpu().numpy()
    recons = recons.cpu().numpy()

    fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))
    for i in range(num_samples):
        axes[0, i].imshow(inputs[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recons[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
    
    if num_samples > 0:
        axes[0, 0].set_ylabel('Original')
        axes[1, 0].set_ylabel('Reconstructed')
        
    plt.tight_layout()
    # Using a relative path for the output file is more portable
    plt.savefig("vae_reconstruction_results.png")
    plt.show()

if __name__ == "__main__":
    main()