import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from model import ConvAutoencoder
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/Autoencoder_data/reconstructed_images_greyscale_reduced"
BATCH_SIZE = 64
LR         = 1e-3
EPOCHS     = 15
LATENT_DIM = 32
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Data Preparation ---
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
])
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# --- Model, Loss & Optimizer ---
model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
rcriterion = nn.MSELoss()

# (Optional) Sobel for edge loss
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

def sobel(x):
    # simple Sobel edge extractor (batch,1,H,W)
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=x.device)
    ky = kx.t()
    kx = kx.view(1,1,3,3)
    ky = ky.view(1,1,3,3)
    ex = F.conv2d(x, kx, padding=1)
    ey = F.conv2d(x, ky, padding=1)
    return torch.sqrt(ex**2 + ey**2)
print("Starting training model 9...")
# --- Training Loop ---
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0.0
    for imgs, _ in train_loader:
        imgs = imgs.to(DEVICE)
        recon, _ = model(imgs)
        loss_recon = rcriterion(recon, imgs)
        loss_edge  = nn.L1Loss()(sobel(recon), sobel(imgs))
        loss = 0.8 * loss_recon + 0.2 * loss_edge

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    recons, inputs = None, None
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(DEVICE)
            recon, _ = model(imgs)
            if recons is None:
                # capture first batch for visualization
                recons = recon.cpu()
                inputs = imgs.cpu()
            loss_recon = rcriterion(recon, imgs)
            loss_edge  = nn.L1Loss()(sobel(recon), sobel(imgs))
            val_loss += (0.8*loss_recon + 0.2*loss_edge).item() * imgs.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch}/{EPOCHS} — Train: {train_loss:.4f}, Val: {val_loss:.4f}")

# --- Visualization of Reconstruction ---

print("Visualizing results...")
# ensure we have at least 10 examples
num_display = min(10, inputs.size(0))
fig, axes = plt.subplots(2, num_display, figsize=(20, 4))
for i in range(num_display):
    axes[0, i].imshow(inputs[i][0], cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(recons[i][0], cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_ylabel('Original')
axes[1, 0].set_ylabel('Reconstructed')
plt.tight_layout()
plt.savefig("/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/model_9/validation_reconstruction.png")
plt.close()
