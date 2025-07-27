# vae_model.py
import torch
import torch.nn as nn

class VAE(nn.Module):
    # Added beta as an initialization parameter
    def __init__(self, latent_dim=32, beta=1.0):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta # Store beta

        # --- Encoder --- (No changes needed here)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # -> 32x42x42
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 64x21x21
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> 128x11x11
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 11 * 11, latent_dim)
        self.fc_logvar = nn.Linear(128 * 11 * 11, latent_dim)

        # --- Decoder ---
        self.fc_decode = nn.Linear(latent_dim, 128 * 11 * 11)

        # --- CORRECTED DECODER ARCHITECTURE ---
        self.decoder = nn.Sequential(
            # Corrected Layer: Removed output_padding=1 to get 21x21 output
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # -> 64x21x21
            nn.ReLU(),
            # This layer now correctly goes from 21x21 -> 42x42
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> 32x42x42
            nn.ReLU(),
            # This layer now correctly goes from 42x42 -> 84x84
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # -> 1x84x84 ✅
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc = self.encoder(x)
        enc_flat = self.flatten(enc)
        mu = self.fc_mu(enc_flat)
        logvar = self.fc_logvar(enc_flat)
        z = self.reparameterize(mu, logvar)
        dec = self.fc_decode(z).view(-1, 128, 11, 11)
        recon = self.decoder(dec)
        return recon, mu, logvar

    # Updated loss function to use the beta parameter
    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Apply the beta weight to the KLD term
        return recon_loss + self.beta * kld, recon_loss, kld