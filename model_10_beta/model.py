# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    """
    A convolutional autoencoder for 84x84 grayscale Pong images, modified
    for less aggressive downsampling and increased latent capacity to better
    capture small objects like the ball.
    """
    def __init__(self, latent_dim=128):
        """
        Initializes the autoencoder.
        Args:
            latent_dim (int): The number of dimensions in the latent space.
                              Increased to 32 to ensure sufficient capacity for
                              details like the ball's position.
        """
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # ---------- Encoder ----------
        # The encoder now downsamples only once to prevent losing small objects.
        self.encoder = nn.Sequential(
            # Input: 1x84x84
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Stays 64x84x84
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Downsamples to 64x42x42

            # Second conv layer without pooling to preserve spatial resolution
            nn.Conv2d(64, 32, kernel_size=3, padding=1), # Becomes 32x42x42
            nn.ReLU(True),
        )

        # Flatten and bottleneck
        # The flattened size is larger due to less pooling.
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(32 * 42 * 42, latent_dim)

        # ---------- Decoder ----------
        self.fc_dec = nn.Linear(latent_dim, 32 * 42 * 42)

        # The decoder reconstructs the image from the 42x42 feature map.
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, 42, 42)),

            # A single transpose convolution upsamples from 42x42 directly to 84x84.
            # It goes from 32 channels to the final 1 grayscale channel.
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()  # Normalize output to [0, 1] for image pixels
        )

    def encode(self, x):
        """
        Encodes an input image into its latent representation.
        Args:
            x (torch.Tensor): The input image tensor (B, C, H, W).
        Returns:
            torch.Tensor: The latent vector `z`.
        """
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        """
        Decodes a latent vector back into a reconstructed image.
        Args:
            z (torch.Tensor): The latent vector.
        Returns:
            torch.Tensor: The reconstructed image.
        """
        x = self.fc_dec(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        """
        The forward pass of the autoencoder.
        Args:
            x (torch.Tensor): The input image tensor.
        Returns:
            torch.Tensor: The reconstructed image.
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon