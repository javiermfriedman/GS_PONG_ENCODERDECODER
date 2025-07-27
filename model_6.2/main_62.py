import os
import torch
from torch.utils.data import DataLoader, random_split # Added random_split
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau  # added for learning rate scheduler
from model62_ import ConvAutoencoderRegularized


# -------- Main Training & Visualization --------
def main():
    # --- 1. Setup and Data Loading ---
    data_dir = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/Autoencoder_data/reconstructed_images_greyscale_reduced"
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])

    print("Loading dataset...")
    # Load the full dataset first
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Split the dataset into training (80%) and validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create separate DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print("Initializing model ---> 6.2 <----- ...")
    model = ConvAutoencoderRegularized() 
    
    # --- CHANGE 1: Lower the learning rate ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) # Original was 1e-3
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    # --- CHANGE 2: Switch to BCEWithLogitsLoss ---
    criterion = torch.nn.BCEWithLogitsLoss() # Original was L1Loss

    
    # Lists to store loss history for plotting
    train_losses = []
    val_losses = []
    num_epochs = 20

    # --- 2. Training & Validation Loop ---
    print("Starting training...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        for imgs, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation
            for imgs, _ in val_loader:
                outputs = model(imgs)
                loss = criterion(outputs, imgs)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        # print current learning rate
        for param_group in optimizer.param_groups:
            print(f"Current LR: {param_group['lr']}")
        
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

    print("Training complete saving model...")
    # model_path = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/model_6/model_lat_16.pth"
    # torch.save(model.state_dict(), model_path)
    # print(f"Model saved to {model_path}")

    # --- 3. Reconstruct and Visualize (using validation data) ---
    print("Reconstructing images from validation set...")
    model.eval()
    
    # Get 10 images from the validation dataset to see performance on unseen data
    inputs = torch.stack([val_dataset[i][0] for i in range(10)])
    with torch.no_grad():
        # Because we removed Sigmoid from the model, we apply it here for visualization
        recons = torch.sigmoid(model(inputs))

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
    plt.savefig("/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/model_6.2/validation_reconstruction.png")
    
    # --- 4. Plot Loss Curves ---
    print("Plotting loss curves...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training & Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/model_6.2/loss_plot.png")
    plt.show()

if __name__ == "__main__":
    main()