import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import ConvAutoencoder
from loss_utils import WeightedPixelwiseMSE  # Assumes this is your loss
import matplotlib.pyplot as plt

# ----- Configuration -----
DATA_DIR = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/Autoencoder_data/reconstructed_images_greyscale/"
BATCH_SIZE = 64
EPOCHS = 5
LATENT_DIM = 128    
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FOREGROUND_WEIGHT=20.0
BACKGROUND_WEIGHT=1.0
FOREGROUND_THRESH=112

ORIGINAL_MODEL_PATH = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/model_10_beta/128_dim_models/model10beta_epochs=15_latent=128_fgweight=10.0.pth"
PATH_TO_SAVE_MODEL = f"/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/model_10_beta/{LATENT_DIM}_dim_models/"
os.makedirs(PATH_TO_SAVE_MODEL, exist_ok=True)

print(f"[MODEL 10 BETA CONTINUE TRAINING]: |foreground weight == {FOREGROUND_WEIGHT}| |epochs == {EPOCHS}| |latent dim == {LATENT_DIM}|")

def main():
    # ----- Transforms -----
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
    ])

    # ----- DataLoader -----
    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    # Split into train and val sets (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # ----- Model, Optimizer -----
    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(ORIGINAL_MODEL_PATH, map_location=DEVICE)) # load the original model
    criterion = WeightedPixelwiseMSE(foreground_weight=FOREGROUND_WEIGHT, background_weight=BACKGROUND_WEIGHT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ----- Training Loop -----
    best_val_loss = float('inf')
    best_model_state = None

    train_loss_history = []
    val_loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * imgs.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)

        # --- 4. Validation Loop ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, imgs)
                total_val_loss += loss.item() * imgs.size(0)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_loss_history.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")  

        # --- Save best model at every epoch ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

            # save best model
            best_model_name = f"fine_tune_2_best_model_epoch{epoch+1}_valloss={avg_val_loss:.6f}.pth"
            torch.save(best_model_state, os.path.join(PATH_TO_SAVE_MODEL, best_model_name))
            show_reconstructions(model, val_loader)

    # # --- 5. Visualize Sample Reconstructions ---
    # show_reconstructions(model, val_loader)

    # --- 5. Plot Loss History ---
    plot_loss_history(train_loss_history, val_loss_history, EPOCHS, LATENT_DIM, FOREGROUND_WEIGHT)

def plot_loss_history(train_loss_history, val_loss_history, EPOCHS, LATENT_DIM, FOREGROUND_WEIGHT):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.legend()
    plt.savefig(os.path.join(PATH_TO_SAVE_MODEL, f'ft_loss_history_epochs={EPOCHS}_latent={LATENT_DIM}_fgweight={FOREGROUND_WEIGHT}.png'))
    plt.show()

def show_reconstructions(model, loader, n=10):
    """
    Plots a grid of n original and reconstructed images from the validation set.
    """
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs[:n].to(DEVICE)
    with torch.no_grad():
        recons = model(imgs)

    imgs = imgs.cpu()
    recons = recons.cpu()

    num_display = min(n, imgs.size(0))
    fig, axes = plt.subplots(2, num_display, figsize=(2*num_display, 4))
    for i in range(num_display):
        axes[0, i].imshow(imgs[i][0], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recons[i][0], cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_SAVE_MODEL, f"ft_model10beta_epochs={EPOCHS}_latent={LATENT_DIM}_fgweight={FOREGROUND_WEIGHT}.png"))
    #plt.show()
        

if __name__ == "__main__":
    main()
